const fs = require('fs');
const path = require('path');

var config = {
    apiKey: "AIzaSyAlkTeBTBPtnVAaOnNmwiwsFVDIWKhfp5M",
    authDomain: "hackathon-mozofest-2019.firebaseapp.com",
    databaseURL: "https://hackathon-mozofest-2019.firebaseio.com",
    storageBucket: 'gs://hackathon-mozofest-2019.appspot.com/'
};

// Prevent double initialization
if (!firebase.apps.length) {
    firebase.initializeApp(config);
}

firebase.auth().onAuthStateChanged(function(user) {
    if (user) {
        console.log("Logged in already : " + user.email);
        M.toast({html:'Welcome back ' + user.email + ' !'});
    }
});

function loadAttendance(day) {
    const csvPath = path.join(__dirname, 'Attendance.csv');
    const studentsPath = path.join(__dirname, 'students.csv');
    // Read students.csv first to build a RegNo->Name map
    fs.readFile(studentsPath, 'utf8', (err, studentsData) => {
        let regNoToName = {};
        if (!err) {
            const studentLines = studentsData.trim().split('\n');
            for (let i = 1; i < studentLines.length; i++) {
                const [reg, name] = studentLines[i].split(',');
                regNoToName[reg] = name;
            }
        }
        fs.readFile(csvPath, 'utf8', (err, data) => {
            if (err) {
                console.error('Error reading attendance:', err);
                document.getElementById('attendanceTable').innerHTML = '<tr><td colspan="3">Could not load attendance.</td></tr>';
                return;
            }
            const lines = data.trim().split('\n');
            const headers = lines[0].split(',');
            const dayIndex = headers.indexOf(day);
            let html = '<table class="striped centered"><thead><tr><th>Reg No</th><th>Name</th><th>Present</th></tr></thead><tbody>';
            for (let i = 1; i < lines.length; i++) {
                const cols = lines[i].split(',');
                let val = cols[dayIndex];
                val = (val === '1') ? '1' : '0';
                let name = regNoToName[cols[0]] || '';
                html += `<tr><td>${cols[0]}</td><td>${name}</td><td>${val}</td></tr>`;
            }
            html += '</tbody></table>';
            document.getElementById('attendanceTable').innerHTML = html;
        });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const elems = document.querySelectorAll('select');
    M.FormSelect.init(elems);
    // Initial load for Day1
    loadAttendance('Day1');
    document.querySelector('select').addEventListener('change', function() {
        const day = 'Day' + this.value;
        loadAttendance(day);
    });
});

function logOut(){
    console.log("Attempting Sign Out");
    firebase.auth().signOut().then(function() {
        console.log("Sign out successful");
        document.location.href = "adminLogin.html";
    }).catch(function(error) {
        console.log("Error signing out");
    });
}

function pyCam(){
    // Get selected day from dropdown
    const selectedDay = document.querySelector('select').value;
    // Call the Python script with the selected day as an argument
    const { spawn } = require('child_process');
    const python = spawn('python', ['py/camcapture.py', selectedDay]);

    python.stdout.on('data', function(data){
        console.log("Python:", data.toString('utf8'));
        M.toast({html: data.toString('utf8')});
        // Reload attendance after update
        loadAttendance('Day' + selectedDay);
    });

    python.stderr.on('data', function(data){
        console.error("Python error:", data.toString('utf8'));
    });

    python.on('close', function(code){
        console.log('Python script exited with code', code);
    });
}

console.log("adminScreen.js ready");