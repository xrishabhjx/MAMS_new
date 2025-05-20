// Initialize Firebase
var config = {
    apiKey: "AIzaSyAlkTeBTBPtnVAaOnNmwiwsFVDIWKhfp5M",
    authDomain: "hackathon-mozofest-2019.firebaseapp.com",
    databaseURL: "https://hackathon-mozofest-2019.firebaseio.com",
    projectId: "hackathon-mozofest-2019",
    storageBucket: "hackathon-mozofest-2019.appspot.com",
    messagingSenderId: "835193922935"
};
firebase.initializeApp(config);

// Clear any signed in users
firebase.auth().signOut().then(function() {
    console.log("Sign out successful");
}).catch(function(error) {
    console.log("Error signing out");
});

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('adminLoginForm').addEventListener('submit', function(e) {
        e.preventDefault();
        foo();
    });
});

function foo() {
    var email = document.getElementById("loginemail").value.trim();
    var password = document.getElementById("loginpassword").value;

    if (!email || !password) {
        M.toast({html: 'Please enter both email and password!'});
        return;
    }

    firebase.auth().signInWithEmailAndPassword(email, password)
        .then(function() {
            var user = firebase.auth().currentUser;
            if (user) {
                M.toast({html: 'Login successful! Redirecting...'});
                setTimeout(function() {
                    document.location.href = "adminScreen.html";
                }, 1000);
            }
        })
        .catch(function(error) {
            console.log(error);
            M.toast({html: 'Invalid credentials! Please try again!'});
        });
}