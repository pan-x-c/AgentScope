var iframe = document.getElementById('workstation-iframe');
var currentUrl = window.location.protocol + '//' + window.location.host + '/workstation';
iframe.src = currentUrl;
console.log(currentUrl)
console.log(123)