# -*- coding: utf-8 -*-
"""The main entry point of the web UI."""
import re
import json
import traceback
import tempfile
import subprocess
import os
from typing import Optional, Tuple
from datetime import datetime

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    Response,
    abort,
    make_response,
)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, join_room, leave_room
from flask_babel import Babel, refresh

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///agentscope.db"
app.config["BABEL_DEFAULT_LOCALE"] = "en"

babel = Babel(app)


def get_locale() -> Optional[str]:
    """
    Determines the best match for the user's locale based on the "locale"
    cookie or the Accept-Language header in the request.
    """
    cookie = request.cookies.get("locale")
    if cookie in ["zh", "en"]:
        return cookie
    return request.accept_languages.best_match(
        app.config.get("BABEL_DEFAULT_LOCALE"),
    )


babel.init_app(app, locale_selector=get_locale)
db = SQLAlchemy(app)
socketio = SocketIO(app)
CORS(app)  # This will enable CORS for all routes

PATH_SAVE = ""


class Run(db.Model):  # type: ignore[name-defined]
    """Run object."""

    id = db.Column(db.String, primary_key=True)
    project = db.Column(db.String)
    name = db.Column(db.String)
    script_path = db.Column(db.String)
    run_dir = db.Column(db.String)
    create_time = db.Column(db.DateTime, default=datetime.now)


class Server(db.Model):  # type: ignore[name-defined]
    """Server object."""

    server_id = db.Column(db.String, primary_key=True)
    server_host = db.Column(db.String)
    server_port = db.Column(db.Integer)


class Message(db.Model):  # type: ignore[name-defined]
    """Message object."""

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String, db.ForeignKey("run.id"), nullable=False)
    name = db.Column(db.String)
    content = db.Column(db.String)
    url = db.Column(db.String)


def get_history_messages(run_id: str) -> list:
    """Interface to get history messages. (Query from database for now)"""
    messages = Message.query.filter_by(run_id=run_id).all()
    return [
        {
            "name": message.name,
            "content": message.content,
            "url": message.url,
        }
        for message in messages
    ]


def get_runs() -> list:
    """Interface to get all runs. (Query from database for now)"""
    runs = Run.query.all()
    return [
        {
            "id": run.id,
            "project": run.project,
            "name": run.name,
            "script_path": run.script_path,
            "run_dir": run.run_dir,
            "create_time": run.create_time.isoformat(),
        }
        for run in runs
    ]


def remove_file_paths(error_trace: str) -> str:
    """
    Remove the real traceback when exception happens.
    """
    path_regex = re.compile(r'File "(.*?)(?=agentscope|app\.py)')
    cleaned_trace = re.sub(path_regex, 'File "[hidden]/', error_trace)

    return cleaned_trace


def convert_to_py(content: str) -> Tuple:
    """
    Convert json config to python code.
    """
    from agentscope.web.workstation.workflow_dag import build_dag

    try:
        cfg = json.loads(content)
        return "True", build_dag(cfg).compile()
    except Exception as e:
        return "False", remove_file_paths(
            f"Error: {e}\n\n" f"Traceback:\n" f"{traceback.format_exc()}",
        )


@app.route("/workstation")
def workstation() -> str:
    """Render the workstation page."""
    return render_template("workstation.html")


@app.route("/api/register/run", methods=["POST"])
def register_run() -> Response:
    """
    Registers a run of an agentscope application.
    The running process will then be displayed as a page.
    """
    # Extract the input data from the request
    data = request.json
    run_id = data.get("run_id")
    project = data.get("project")
    name = data.get("name")
    run_dir = data.get("run_dir")
    # check if the run_id is already in the database
    if Run.query.filter_by(id=run_id).first():
        print(f"Run id {run_id} already exists.")
        abort(400, f"RUN_ID {run_id} already exists")
    db.session.add(
        Run(
            id=run_id,
            project=project,
            name=name,
            run_dir=run_dir,
        ),
    )
    db.session.commit()
    print(f"Register Run id {run_id}.")
    return jsonify(status="ok", msg="")


@app.route("/api/register/server", methods=["POST"])
def register_server() -> Response:
    """
    Registers an agent server.
    """
    data = request.json
    server_id = data.get("server_id")
    host = data.get("host")
    port = data.get("port")
    run_dir = data.get("run_dir")

    if Server.query.filter_by(server_id=server_id).first():
        return jsonify(status="error", msg="server_id already exists")
    else:
        db.session.add(
            Server(
                server_id=server_id,
                server_host=host,
                server_port=port,
                run_dir=run_dir,
            ),
        )
        return jsonify(status="ok", msg="")


@app.route("/api/message/put", methods=["POST"])
def put_message() -> Response:
    """
    Used by the application to speak a message to the Hub.
    """
    data = request.json
    run_id = data["run_id"]
    name = data["name"]
    content = data["content"]
    url = data.get("url", None)
    try:
        new_message = Message(
            run_id=run_id,
            name=name,
            content=content,
            url=url,
        )
        db.session.add(new_message)
        db.session.commit()
    except Exception as e:
        print(e)
        abort(400, "Fail to put message")
    socketio.emit(
        "display_message",
        {
            "run_id": run_id,
            "name": name,
            "content": content,
            "url": url,
        },
        room=run_id,
    )
    return jsonify(status="ok", msg="")


@app.route("/api/messages/<run_id>", methods=["GET"])
def get_messages(run_id: str) -> list:
    """Get the history messages of specific run_id."""
    return get_history_messages(run_id=run_id)


@app.route("/api/runs", methods=["GET"])
def get_all_runs() -> list:
    """Get all runs."""
    return get_runs()


@app.route("/studio/<run_id>", methods=["GET"])
def studio_page(run_id: str) -> str:
    """Studio page."""
    if Run.query.filter_by(id=run_id).first() is None:
        return jsonify(status="error", msg="run_id not exists")
    messages = Message.query.filter_by(run_id=run_id).all()
    return render_template("chat.html", messages=messages, run_id=run_id)


@app.route("/getProjects", methods=["GET"])
def get_projects() -> Response:
    """Get all the projects in the runs directory."""
    cfgs = []
    for run_dir in os.listdir(PATH_SAVE):
        print(run_dir)
        path_cfg = os.path.join(PATH_SAVE, run_dir, ".config")
        if os.path.exists(path_cfg):
            with open(path_cfg, "r", encoding="utf-8") as file:
                cfg = json.load(file)
                cfg["dir"] = run_dir
                cfgs.append(cfg)

    # Filter the same projects
    project_names = list({_["project"] for _ in cfgs})

    return jsonify(
        {
            "names": project_names,
            "runs": cfgs,
        },
    )


@app.route("/convert-to-py", methods=["POST"])
def convert_config_to_py() -> Response:
    """
    Convert json config to python code and send back.
    """
    content = request.json.get("data")
    status, py_code = convert_to_py(content)
    return jsonify(py_code=py_code, is_success=status)


@app.route("/convert-to-py-and-run", methods=["POST"])
def convert_config_to_py_and_run() -> Response:
    """
    Convert json config to python code and run.
    """
    content = request.json.get("data")
    status, py_code = convert_to_py(content)

    if status == "True":
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".py",
                mode="w+t",
            ) as tmp:
                tmp.write(py_code)
                tmp.flush()
                # TODO: use the latest implementation
                subprocess.Popen(
                    ["python", tmp.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
        except Exception as e:
            status, py_code = "False", remove_file_paths(
                f"Error: {e}\n\n" f"Traceback:\n" f"{traceback.format_exc()}",
            )
    return jsonify(py_code=py_code, is_success=status)


@app.route("/read-examples", methods=["POST"])
def read_examples() -> Response:
    """
    Read tutorial examples from local file.
    """
    lang = request.json.get("lang")
    file_index = request.json.get("data")

    # TODO: might need to fix path
    if not os.path.exists(
        os.path.join(
            "workstation",
            "tutorials",
            f"{lang}{file_index}.json",
        ),
    ):
        lang = "en"

    with open(
        os.path.join(
            "workstation",
            "tutorials",
            f"{lang}{file_index}.json",
        ),
        "r",
        encoding="utf-8",
    ) as jf:
        data = json.load(jf)
    return jsonify(json=data)


@app.route("/set_locale")
def set_locale() -> Response:
    """
    Sets the user's preferred language in a cookie based on the query
    parameter "language".

    Supports setting the language preference to either English ("en") or
    Chinese ("zh"). If a supported language is specified, it sets a
    corresponding cookie and returns a JSON response with a success message.
    For unsupported or missing language preferences, it returns a JSON
    response indicating success without setting a language preference cookie.
    """
    lang = request.args.get("language")
    response = make_response(jsonify(message=lang))
    if lang == "en":
        refresh()
        response.set_cookie("locale", "en")
        return response

    if lang == "zh":
        refresh()
        response.set_cookie("locale", "zh")
        return response

    return jsonify({"data": "success"})


@app.route("/")
def home() -> str:
    """Render the home page."""
    return render_template("home.html")


@app.route("/run/<run_dir>")
def run_detail(run_dir: str) -> str:
    """Render the run detail page."""
    path_run = os.path.join(PATH_SAVE, run_dir)

    # Find the logging and chat file by suffix
    path_log = os.path.join(path_run, "logging.log")
    path_dialog = os.path.join(path_run, "logging.chat")

    if os.path.exists(path_log):
        with open(path_log, "r", encoding="utf-8") as file:
            logging_content = ["".join(file.readlines())]
    else:
        logging_content = None

    if os.path.exists(path_dialog):
        with open(path_dialog, "r", encoding="utf-8") as file:
            dialog_content = file.readlines()
        dialog_content = [json.loads(_) for _ in dialog_content]
    else:
        dialog_content = []

    path_cfg = os.path.join(PATH_SAVE, run_dir, ".config")
    if os.path.exists(path_cfg):
        with open(path_cfg, "r", encoding="utf-8") as file:
            cfg = json.load(file)
    else:
        cfg = {
            "project": "-",
            "name": "-",
            "id": "-",
            "timestamp": "-",
        }

    logging_and_dialog = {
        "config": cfg,
        "logging": logging_content,
        "dialog": dialog_content,
    }

    return render_template("run.html", runInfo=logging_and_dialog)


@socketio.on("user_input")
def user_input(data: dict) -> None:
    """Get user input and send to the agent"""
    run_id = data["run_id"]
    content = data["content"]
    url = data.get("url", None)
    socketio.emit(
        "fetch_user_input",
        {
            "run_id": run_id,
            "content": content,
            "url": url,
        },
        room=run_id,
    )


@socketio.on("connect")
def on_connect() -> None:
    """Execute when a client is connected."""
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect() -> None:
    """Execute when a client is disconnected."""
    print("Client disconnected")


@socketio.on("join")
def on_join(data: dict) -> None:
    """Join a websocket room"""
    run_id = data["run_id"]
    join_room(run_id)


@socketio.on("leave")
def on_leave(data: dict) -> None:
    """Leave a websocket room"""
    run_id = data["run_id"]
    leave_room(run_id)


def init(
    path_save: str,
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
) -> None:
    """Start the web UI."""
    global PATH_SAVE

    if not os.path.exists(path_save):
        raise FileNotFoundError(f"The path {path_save} does not exist.")
    with app.app_context():
        db.create_all()
    PATH_SAVE = path_save
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
