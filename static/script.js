document.getElementById('upload-form').onsubmit = async function (e) {

    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = "Predicting...";

    try {
        const response = await fetch(form.action, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        resultDiv.innerHTML = `
                    <h3>Predicted Class: <strong>${data.prediction}</strong></h3>
                    <p>Confidence: ${data.confidence}</p>
                `;

    } catch (error) {
        resultDiv.innerHTML = `An error occurred: ${error.message}`;
    }
};



