document.addEventListener("DOMContentLoaded", () => {
  const charElement = document.getElementById("chart");

  if (!charElement) {
    console.error("No chart element found");
    return;
  }

  const ctx = charElement.getContext("2d");
  const gradient = ctx.createLinearGradient(0, -10, 0, 100);

  gradient.addColorStop(0, "rgba(255, 0, 0, 1)");
  gradient.addColorStop(1, "rgba(255, 99, 132, 1)");

  const forecastItem = document.querySelectorAll(".forecast-item");
  const temps = [];
  const times = [];

  forecastItem.forEach((item) => {
    const time = item.querySelector(".forecast-time").textContent;
    const temp = item.querySelector(".forecast-tempValue").textContent;
    const hum = item.querySelector(".forecast-humidityValue").textContent;

    if (time && temp && hum) {
      temps.push(temp);
      times.push(time);
    }
  });

  // Ensure all values are valid before using them to create the chart
  if (temps.length === 0 || times.length === 0) {
    console.error("No valid data found");
    return;
  }

  new Chart(ctx, {
    type: "line",
    data: {
      labels: times,
      datasets: [
        {
          label: "Celcius Degrees",
          data: temps,
          borderColor: gradient,
          tension: 0.4,
          pointRadius: 2,
        },
      ],
    },
    options: {
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          display: false,
          grid: {
            drawOnChartArea: false,
          },
        },
        y: {
          display: false,
          grid: {
            drawOnChartArea: false,
          },
        },
      },
      animation: {
        duration: 7500,
      },
    },
  });
});
