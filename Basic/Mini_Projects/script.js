const analyzeBtn = document.getElementById("analyzeBtn");

analyzeBtn.addEventListener("click", async () => {
    const description = document.getElementById("description").value;
    const result = document.getElementById("result");

    if (!description.trim()) {
        result.innerText = "Please enter a movie description.";
        return;
    }

    result.innerText = "Analyzing...";

    try {
        const response = await fetch("http://localhost:8000/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                description: description
            })
        });

        const data = await response.json();

        result.innerText = data.response;
    } catch (error) {
        result.innerText = "Error connecting to the backend.";
    }
});