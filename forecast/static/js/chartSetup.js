document.addEventListener("DOMContentLoaded", () => {
  // Find the chart element
  const chartElement = document.getElementById("chart");

  // Check if the chart element exists
  if (!chartElement) {
    console.error("No chart element found");
    return; // Exit if no chart element is present
  }

  const ctx = chartElement.getContext("2d"); // Initialize chart context
  const gradient = ctx.createLinearGradient(0, -10, 0, 100); // Create gradient for the chart line

  // Define gradient colors (top to bottom)
  gradient.addColorStop(0, "rgba(255, 0, 0, 1)"); // Start with red
  gradient.addColorStop(1, "rgba(255, 99, 132, 1)"); // Blend into pink

  // Get forecast data from HTML elements with the class `forecast-item`
  const forecastItems = document.querySelectorAll(".forecast-item");
  const temps = []; // Array for temperatures
  const times = []; // Array for time labels

  // Loop over forecast items to extract time and temperature
  forecastItems.forEach((item) => {
    const time = item.querySelector(".forecast-time").textContent; // Time for each forecast
    const temp = item.querySelector(".forecast-tempValue").textContent; // Temperature for each forecast
    const hum = item.querySelector(".forecast-humidityValue").textContent; // Humidity, though unused here

    // Add valid time and temperature data to respective arrays
    if (time && temp && hum) {
      temps.push(temp); // Add temperature to `temps`
      times.push(time); // Add time to `times`
    }
  });

  // Check if there is data to plot
  if (temps.length === 0 || times.length === 0) {
    console.error("No valid data found");
    return;
  }

  // Create and display the chart
  new Chart(ctx, {
    type: "line", // Line chart type
    data: {
      labels: times, // Time labels on x-axis
      datasets: [
        {
          label: "Celcius Degrees", // Dataset label
          data: temps, // Temperature data
          borderColor: gradient, // Apply gradient as line color
          tension: 0.4, // Smooth the line
          pointRadius: 2, // Small points on line
        },
      ],
    },
    options: {
      plugins: {
        legend: { display: false }, // Hide legend
      },
      scales: {
        x: {
          display: false, // Hide x-axis labels
          grid: { drawOnChartArea: false }, // Remove grid
        },
        y: {
          display: false, // Hide y-axis labels
          grid: { drawOnChartArea: false }, // Remove grid
        },
      },
      animation: { duration: 7500 }, // Slow down animation
    },
  });
});
