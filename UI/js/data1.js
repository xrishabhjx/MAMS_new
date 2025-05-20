var config = {
    apiKey: "AIzaSyAlkTeBTBPtnVAaOnNmwiwsFVDIWKhfp5M",
    authDomain: "hackathon-mozofest-2019.firebaseapp.com",
    databaseURL: "https://hackathon-mozofest-2019.firebaseio.com",
    storageBucket: 'gs://hackathon-mozofest-2019.appspot.com/'
};

if (!firebase.apps.length) {
    firebase.initializeApp(config);
}

var database = firebase.database();

var starCountRef = database.ref('Students');
starCountRef.on('value', function(snapshot) {
    var response = snapshot.val();
    console.log(response);
    // Replace 'studentRegNoDisplay' with your actual element id
    var displayElem = document.getElementById('studentRegNoDisplay');
    if (displayElem) {
        if (response && response.RA1711003010350) {
            displayElem.innerHTML = response.RA1711003010350.RegNo;
        } else {
            displayElem.innerHTML = 'No data found';
        }
    }
});