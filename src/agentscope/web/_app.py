# -*- coding: utf-8 -*-
"""The main entry point of the web UI."""
import json
import os
from multiprocessing import Queue

from loguru import logger
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_socketio import SocketIO


app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)  # This will enable CORS for all routes


PATH_SAVE = ""


def init_uid_queues() -> dict:
    """Initializes and returns a dictionary of user-specific queues."""
    return {
        "glb_queue_chat_msg": Queue(),
        "glb_queue_user_input": Queue(),
        "glb_queue_reset_msg": Queue(),
    }


glb_uid_dict = {}


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


@app.route("/")
def home() -> str:
    """Render the home page."""
    return render_template("home.html")


@app.route("/studio/run/<run_id>", methods=["GET"])
def studio(run_id: str) -> str:
    """Render the studio"""
    return render_template(
        "studio.html", run_id=run_id, studio_ip="127.0.0.1", studio_port=7860
    )


@app.route("/api/register/run", methods=["POST"])
def register_run() -> Response:
    """Register an application run."""
    global glb_uid_dict
    data = request.get_json()
    run_id = data["run_id"]
    if run_id not in glb_uid_dict:
        glb_uid_dict[run_id] = {
            "glb_queue_chat_msg": Queue(),
            "glb_queue_user_input": Queue(),
            "glb_queue_reset_msg": Queue(),
        }
        return {
            "status": "success",
            "msg": f"run_id {run_id} registered",
            "uid": run_id,
        }
    else:
        logger.warning(f"RUN_ID [{run_id}] already exists.")
        return {
            "status": "error",
            "msg": f"run_id {run_id} already exists",
            "uid": run_id,
        }


@app.route("/api/message/send/chat", methods=["POST"])
def send_msg() -> Response:
    """Send a message to the chat."""
    global glb_uid_dict
    data = request.get_json()
    run_id = data["run_id"]
    msg = data["msg"]
    if run_id in glb_uid_dict:
        glb_uid_dict[run_id]["glb_queue_chat_msg"].put(msg)
        return jsonify({"status": "success"})
    else:
        return jsonify(
            {"status": "error", "msg": f"run_id {run_id} not exists"}
        )


@app.route("/api/message/send/player", methods=["POST"])
def send_player_input() -> Response:
    """Send a message to the chat."""
    global glb_uid_dict
    data = request.get_json()
    run_id = data["run_id"]
    msg = data["msg"]
    if run_id in glb_uid_dict:
        glb_uid_dict[run_id]["glb_queue_user_input"].put([None, msg])
        return jsonify({"status": "success"})
    else:
        return jsonify(
            {"status": "error", "msg": f"run_id {run_id} not exists"}
        )


@app.route("/api/message/get/chat/<run_id>", methods=["GET"])
def get_chat_msg(run_id: str) -> Response:
    """Get chat messages from the queue."""
    global glb_uid_dict
    glb_queue_chat_msg = glb_uid_dict[run_id]["glb_queue_chat_msg"]
    if not glb_queue_chat_msg.empty():
        line = glb_queue_chat_msg.get(block=False)
        if line is not None:
            return {"status": "success", "msg": line}
    return {"status": "success", "msg": []}


@app.route("/api/message/get/player/<run_id>", methods=["GET"])
def get_player_input(run_id: str) -> Response:
    """Get player input from the queue."""
    global glb_uid_dict
    glb_queue_player_input = glb_uid_dict[run_id]["glb_queue_user_input"]
    content = glb_queue_player_input.get(block=True)[1]
    return {"status": "success", "msg": content}


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


@socketio.on("connect")
def on_connect() -> None:
    """Execute when a client is connected."""
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect() -> None:
    """Execute when a client is disconnected."""
    print("Client disconnected")


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

    PATH_SAVE = path_save
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True,
    )
