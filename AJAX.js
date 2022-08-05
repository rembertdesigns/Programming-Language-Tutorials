Asynchronous JAvascript and Xml 
  -API communications
    -by Richard
    
    

// AJAX GET func
// target URL and callback func as parameters (callback runs if success)
function ajaxGet(url, callback) {
  var req = new XMLHttpRequest();
  req.open("GET", url);
  req.addEventListener("load", function () {
    if (req.status >= 200 && req.status < 400) {
      // callback with request answer
      callback(req.responseText);
    } else {
      console.error(req.status + " " + req.statusText + " " + url);
    }
  });
  req.addEventListener("error", function () {
    console.error("Network error with URL : " + url);
  });
  req.send(null);
}




// AJAX POST func
// target URL, data to send and callback func as parameters (callback runs if success)
function ajaxPost(url, data, callback, isJson) { // isJson checks if data is json format
  var req = new XMLHttpRequest();
  req.open("POST", url);
  req.addEventListener("load", function () {
    if (req.status >= 200 && req.status < 400) {
      // callback with request answer
      callback(req.responseText);
    } else {
      console.error(req.status + " " + req.statusText + " " + url);
    }
  });
  req.addEventListener("error", function () {
    console.error("Network error with URL : " + url);
  });
  if (isJson) {
    // request content is JSON
    req.setRequestHeader("Content-Type", "application/json");
    // turns data into JSON before sending
    data = JSON.stringify(data);
  }
  req.send(data);
}