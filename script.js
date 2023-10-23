let numDays;
let dailyData = [];
let model;
let averageDuration = 0;
let userInfo = {}; // Create an object to store user information

function setupTable() {
    numDays = parseInt(document.getElementById("days").value);
    let tableHtml = "";
    for (let i = 0; i < numDays; i++) {
        tableHtml += `
            <tr>
                <td>${i + 1}</td>
                <td><input type="number" id="duration-${i}" min="0" max="24"></td>
                <td><input type="number" id="activity-${i}" min="1" max="10"></td>
                <td><input type="number" id="stress-${i}" min="1" max="10"></td>
                <td><input type="number" id="quality-${i}" min="1" max="10"></td>
            </tr>
        `;
    }
    document.getElementById("table-body").innerHTML = tableHtml;

    // Store user information
    userInfo.age = parseInt(document.getElementById("age").value);
    userInfo.gender = document.getElementById("gender").value;
    userInfo.name = document.getElementById("name").value;
    userInfo.email = document.getElementById("email").value;
    userInfo.numDays = numDays;
}
function trainModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [3] }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    dailyData = [];
    for (let i = 0; i < numDays; i++) {
        const duration = parseFloat(document.getElementById(`duration-${i}`).value);
        const activity = parseFloat(document.getElementById(`activity-${i}`).value);
        const stress = parseFloat(document.getElementById(`stress-${i}`).value);
        const quality = parseFloat(document.getElementById(`quality-${i}`).value);
        dailyData.push({ duration, activity, stress, quality });
    }

    const trainData = dailyData.map(entry => {
        return {
            x: [entry.duration, entry.activity, entry.stress],
            y: [entry.quality]
        };
    });

    const xs = tf.tensor(trainData.map(entry => entry.x));
    const ys = tf.tensor(trainData.map(entry => entry.y));

    model.fit(xs, ys, {
        epochs: 200,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    });

    // Calculate the average duration
    averageDuration = dailyData.reduce((acc, entry) => acc + entry.duration, 0) / numDays;
}

function askSleepDisorderQuestions() {
    document.getElementById("diagnosis-form").style.display = "block";
}

function processDiagnosis() {
    const snoring = document.querySelector('input[name="snoring"]:checked').value;
    const awakenings = document.querySelector('input[name="awakenings"]:checked').value;

    if (snoring === "yes") {
        document.getElementById("diagnosis-message").textContent = "You may have sleep apnea. Please consult a healthcare professional.";
    } else if (awakenings === "yes") {
        document.getElementById("diagnosis-message").textContent = "You may have insomnia. Consider improving your sleep habits and consult a healthcare professional if needed.";
    } else {
        document.getElementById("diagnosis-message").textContent = "Your sleep duration is below recommended levels. Consider improving your sleep habits.";
    }

    // Hide the form
    document.getElementById("diagnosis-form").style.display = "none";
}

function predictQualityForFutureDay() {
    const duration = parseFloat(document.getElementById("predict-duration").value);
    const activity = parseFloat(document.getElementById("predict-activity").value);
    const stress = parseFloat(document.getElementById("predict-stress").value);

    if (validateInput(duration, activity, stress)) {
        const inputTensor = tf.tensor2d([[duration, activity, stress]]);
        const predictedQuality = model.predict(inputTensor);
        const predictedQualityValue = predictedQuality.dataSync()[0];

        document.getElementById("predicted-quality").textContent = `Predicted Sleep Quality: ${predictedQualityValue.toFixed(2)}`;

        // Suggest possible sleep disorders for a future day
        if (averageDuration > 7) {
            document.getElementById("healthy").textContent = "Your sleep is Healthy ";
        }
        if (averageDuration <= 5) {
            askSleepDisorderQuestions();
        }

    } else {
        alert("Please enter valid values within the specified ranges.");
    }
}


function validateInput(duration, activity, stress) {
    const isValidDuration = duration >= 0 && duration <= 24;
    const isValidActivity = activity >= 1 && activity <= 10;
    const isValidStress = stress >= 1 && stress <= 10;

    return isValidDuration && isValidActivity && isValidStress;
}
