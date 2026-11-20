// SmartCanteen Full Web Application Client Controller

document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  loadAnalytics();
  loadDatasetInfo();
  loadModelInfo();
  setupPredictionForm();
  setupSimulator();
  setupTraining();
  setupUploadAndReset();
  setupNewFoodDataEntry();
  loadHistoryTable();
});

// Sidebar Navigation
function setupNavigation() {
  const links = document.querySelectorAll(".nav-link");
  const tabs = document.querySelectorAll(".tab-content");
  const titleHeader = document.getElementById("page-title-header");

  links.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("data-tab");

      links.forEach(l => l.classList.remove("active"));
      tabs.forEach(t => t.classList.remove("active"));

      link.classList.add("active");
      const targetTab = document.getElementById(targetId);
      if (targetTab) {
        targetTab.classList.add("active");
      }

      if (titleHeader) {
        titleHeader.innerHTML = link.innerHTML;
      }

      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });
}

// Fetch & Render Analytics Data
async function loadAnalytics() {
  try {
    const res = await fetch("/api/analytics");
    const data = await res.json();
    if (!data.success) return;

    document.getElementById("kpi-prepared").textContent = data.kpis.total_prepared.toLocaleString();
    document.getElementById("kpi-sold").textContent = data.kpis.total_sold.toLocaleString();
    document.getElementById("kpi-leftover").textContent = data.kpis.total_leftover.toLocaleString();
    document.getElementById("kpi-waste-pct").textContent = data.kpis.waste_pct + "%";
    document.getElementById("kpi-avg-daily").textContent = data.kpis.avg_daily.toLocaleString() + " meals";
    document.getElementById("kpi-top-item").textContent = data.kpis.top_item;
    document.getElementById("kpi-top-meal").textContent = data.kpis.top_meal;
    document.getElementById("kpi-top-day").textContent = data.kpis.top_day;

    renderCharts(data.charts);
  } catch (err) {
    console.error("Error loading analytics:", err);
  }
}

function renderCharts(charts) {
  // Chart 1: Daily Volume Trend
  const trendTrace1 = {
    x: charts.daily_dates,
    y: charts.daily_prepared,
    type: 'scatter',
    mode: 'lines',
    name: 'Prepared',
    line: { color: '#ff9800', width: 2 }
  };
  const trendTrace2 = {
    x: charts.daily_dates,
    y: charts.daily_sold,
    type: 'scatter',
    mode: 'lines',
    name: 'Sold (Demand)',
    line: { color: '#2e7d32', width: 2.5 }
  };
  Plotly.newPlot('chart-daily-trend', [trendTrace1, trendTrace2], {
    margin: { t: 30, r: 20, l: 45, b: 40 },
    hovermode: 'x unified',
    legend: { orientation: 'h', y: 1.15 }
  }, { responsive: true, displayModeBar: false });

  // Chart 2: Day of Week Bar
  const dowTrace = {
    x: charts.dow_labels,
    y: charts.dow_values,
    type: 'bar',
    marker: { color: '#4caf50' }
  };
  Plotly.newPlot('chart-dow', [dowTrace], {
    margin: { t: 20, r: 20, l: 45, b: 40 }
  }, { responsive: true, displayModeBar: false });

  // Chart 3: Meal Share Donut
  const mealTrace = {
    labels: charts.meal_labels,
    values: charts.meal_values,
    type: 'pie',
    hole: 0.45,
    marker: { colors: ['#81c784', '#2e7d32', '#ffb74d'] }
  };
  Plotly.newPlot('chart-meal-share', [mealTrace], {
    margin: { t: 20, r: 20, l: 20, b: 20 }
  }, { responsive: true, displayModeBar: false });

  // Chart 4: Food Category Horizontal Bar
  if (charts.cat_labels && charts.cat_values) {
    const catTrace = {
      x: charts.cat_values,
      y: charts.cat_labels,
      type: 'bar',
      orientation: 'h',
      marker: { color: '#1b5e20' }
    };
    Plotly.newPlot('chart-cat-demand', [catTrace], {
      margin: { t: 20, r: 20, l: 110, b: 40 }
    }, { responsive: true, displayModeBar: false });
  }

  // Chart 5: Top Items Bar
  const topItemsTrace = {
    x: charts.top_item_labels,
    y: charts.top_item_values,
    type: 'bar',
    marker: { color: '#2e7d32' }
  };
  Plotly.newPlot('chart-top-items', [topItemsTrace], {
    margin: { t: 20, r: 20, l: 45, b: 85 },
    xaxis: { tickangle: -35 }
  }, { responsive: true, displayModeBar: false });

  // Chart 6: Surplus Leftover Bar
  const wasteTrace = {
    x: charts.waste_item_labels,
    y: charts.waste_item_values,
    type: 'bar',
    marker: { color: '#e53935' }
  };
  Plotly.newPlot('chart-waste-items', [wasteTrace], {
    margin: { t: 20, r: 20, l: 45, b: 85 },
    xaxis: { tickangle: -35 }
  }, { responsive: true, displayModeBar: false });

  // Chart 7: Academic Impact (Normal vs Exam vs Holiday)
  if (charts.period_labels) {
    const pTrace1 = {
      x: charts.period_labels,
      y: charts.period_prepared,
      name: 'Prepared',
      type: 'bar',
      marker: { color: '#ffa000' }
    };
    const pTrace2 = {
      x: charts.period_labels,
      y: charts.period_sold,
      name: 'Sold',
      type: 'bar',
      marker: { color: '#2e7d32' }
    };
    const pTrace3 = {
      x: charts.period_labels,
      y: charts.period_leftover,
      name: 'Leftover',
      type: 'bar',
      marker: { color: '#e53935' }
    };
    Plotly.newPlot('chart-academic-impact', [pTrace1, pTrace2, pTrace3], {
      barmode: 'group',
      margin: { t: 20, r: 20, l: 45, b: 40 },
      legend: { orientation: 'h', y: 1.15 }
    }, { responsive: true, displayModeBar: false });
  }
}

// Dataset Info Fetcher
async function loadDatasetInfo() {
  try {
    const res = await fetch("/api/dataset-info");
    const data = await res.json();
    if (!data.success) return;

    document.getElementById("ds-rows").textContent = data.total_rows.toLocaleString();
    document.getElementById("ds-cols").textContent = data.total_cols;
    document.getElementById("ds-items").textContent = data.unique_items;
    document.getElementById("ds-range").textContent = data.date_range;
    document.getElementById("sidebar-ds-count").textContent = data.total_rows.toLocaleString() + " records";

    // Populate food item dropdown in prediction tab
    const selectFood = document.getElementById("pred-food-item");
    if (selectFood && data.food_items) {
      selectFood.innerHTML = "";
      data.food_items.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        selectFood.appendChild(opt);
      });
    }

    // Populate Category dropdown in prediction tab
    const selectCat = document.getElementById("pred-category");
    if (selectCat && data.categories) {
      selectCat.innerHTML = "";
      data.categories.forEach(cat => {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.textContent = cat;
        selectCat.appendChild(opt);
      });
    }

    // Render Table Preview
    const tableBody = document.getElementById("preview-table-body");
    if (tableBody && data.preview_rows) {
      tableBody.innerHTML = "";
      data.preview_rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${row.Date}</td>
          <td>${row.Day_of_Week}</td>
          <td>${row.Meal_Type}</td>
          <td><b>${row.Food_Item}</b></td>
          <td>${row.Food_Category}</td>
          <td>${row.Quantity_Prepared}</td>
          <td>${row.Quantity_Sold}</td>
          <td>${row.Leftover_Quantity}</td>
          <td>${row.Holiday_Flag ? 'Yes' : 'No'}</td>
          <td>${row.Exam_Week_Flag ? 'Yes' : 'No'}</td>
          <td>${row.Expected_Attendance}</td>
          <td>₹${row.Price_per_Item}</td>
        `;
        tableBody.appendChild(tr);
      });
    }

    // Render Descriptive Statistics Table
    const statsBody = document.getElementById("stats-table-body");
    if (statsBody && data.stats_rows) {
      statsBody.innerHTML = "";
      data.stats_rows.forEach(s => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><b>${s.feature}</b></td>
          <td>${s.mean}</td>
          <td>${s.std}</td>
          <td>${s.min}</td>
          <td>${s.max}</td>
        `;
        statsBody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Error loading dataset preview:", err);
  }
}

// User Input Section: Add Food / Sales Record
function setupNewFoodDataEntry() {
  const form = document.getElementById("new-food-data-form");
  if (!form) return;

  const dateInput = document.getElementById("entry-date");
  if (dateInput) {
    dateInput.value = new Date().toISOString().split("T")[0];
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const submitBtn = form.querySelector("button[type='submit']");
    submitBtn.textContent = "Saving Record...";
    submitBtn.disabled = true;

    const payload = {
      Date: document.getElementById("entry-date").value,
      Meal_Type: document.getElementById("entry-meal-type").value,
      Food_Item: document.getElementById("entry-food-item").value.trim(),
      Food_Category: document.getElementById("entry-category").value.trim(),
      Quantity_Prepared: parseInt(document.getElementById("entry-prepared").value) || 0,
      Quantity_Sold: parseInt(document.getElementById("entry-sold").value) || 0,
      Price_per_Item: parseFloat(document.getElementById("entry-price").value) || 50.0,
      Expected_Attendance: parseInt(document.getElementById("entry-attendance").value) || 1000,
      Holiday_Flag: parseInt(document.getElementById("entry-holiday").value) || 0,
      Exam_Week_Flag: parseInt(document.getElementById("entry-exam").value) || 0,
      Temperature_C: parseFloat(document.getElementById("entry-temp").value) || 28.0,
      Rainfall_mm: parseFloat(document.getElementById("entry-rain").value) || 0.0
    };

    try {
      const res = await fetch("/api/add-food-data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      submitBtn.textContent = "💾 Save & Append Record to Dataset";
      submitBtn.disabled = false;

      if (data.success) {
        const feedback = document.getElementById("entry-feedback");
        const msg = document.getElementById("entry-feedback-msg");
        feedback.style.display = "block";
        msg.textContent = `${data.message} Leftover calculated: ${payload.Quantity_Prepared - payload.Quantity_Sold} portions.`;
        
        // Refresh tables and analytics
        loadDatasetInfo();
        loadAnalytics();
        
        // Clear input dish name
        document.getElementById("entry-food-item").value = "";
      } else {
        alert("Error saving record: " + data.error);
      }
    } catch (err) {
      alert("Submission error: " + err);
      submitBtn.textContent = "💾 Save & Append Record to Dataset";
      submitBtn.disabled = false;
    }
  });
}

// Load Model Info and Feature Importance
async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    if (!data.trained) return;

    document.getElementById("sidebar-model-name").textContent = data.best_model_name;
    document.getElementById("active-model-title").textContent = data.best_model_name;

    if (data.leaderboard && data.leaderboard.length > 0) {
      const lbBody = document.getElementById("leaderboard-body");
      if (lbBody) {
        lbBody.innerHTML = "";
        data.leaderboard.forEach((m, idx) => {
          const tr = document.createElement("tr");
          const isBest = idx === 0;
          if (isBest) tr.style.backgroundColor = "#f1f8e9";
          tr.innerHTML = `
            <td><b>${m.Model}</b></td>
            <td>${m.MAE}</td>
            <td>${m.RMSE}</td>
            <td><b>${(m.R2_Score * 100).toFixed(1)}%</b> (${m.R2_Score})</td>
            <td>${isBest ? '<span class="badge-green">Champion (Active)</span>' : '<span class="badge-yellow">Candidate</span>'}</td>
          `;
          lbBody.appendChild(tr);
        });
      }
    }

    if (data.feature_labels && data.feature_labels.length > 0) {
      const fiTrace = {
        x: data.feature_values.slice().reverse(),
        y: data.feature_labels.slice().reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#2e7d32' }
      };
      Plotly.newPlot('chart-feature-importance', [fiTrace], {
        margin: { t: 20, r: 20, l: 150, b: 40 }
      }, { responsive: true, displayModeBar: false });
    }
  } catch (err) {
    console.error("Error loading model info:", err);
  }
}

// Prediction Form Logic
function setupPredictionForm() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  const dateInput = document.getElementById("pred-date");
  if (dateInput) {
    const tom = new Date();
    tom.setDate(tom.getDate() + 1);
    dateInput.value = tom.toISOString().split("T")[0];
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = form.querySelector("button[type='submit']");
    btn.textContent = "Forecasting with ML...";
    btn.disabled = true;

    const payload = {
      Date: document.getElementById("pred-date").value,
      Meal_Type: document.getElementById("pred-meal-type").value,
      Food_Item: document.getElementById("pred-food-item").value,
      Food_Category: document.getElementById("pred-category").value,
      Expected_Attendance: parseFloat(document.getElementById("pred-attendance").value),
      Holiday_Flag: parseInt(document.getElementById("pred-holiday").value),
      Exam_Week_Flag: parseInt(document.getElementById("pred-exam").value),
      Temperature_C: parseFloat(document.getElementById("pred-temp").value),
      Rainfall_mm: parseFloat(document.getElementById("pred-rain").value),
      Price_per_Item: parseFloat(document.getElementById("pred-price").value),
      Usual_Preparation: parseInt(document.getElementById("pred-usual").value),
      Sales_Lag_1: parseFloat(document.getElementById("pred-lag1").value),
      Sales_Lag_7: parseFloat(document.getElementById("pred-lag7").value),
      Moving_Avg_7: parseFloat(document.getElementById("pred-avg7").value),
      Moving_Avg_14: parseFloat(document.getElementById("pred-avg14").value),
      Weekday_Avg_Sales: parseFloat(document.getElementById("pred-dow-avg").value)
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await res.json();
      btn.textContent = "🔮 Forecast Demand & Kitchen Recommendation";
      btn.disabled = false;

      if (result.success) {
        document.getElementById("pred-results-card").style.display = "block";
        document.getElementById("res-predicted").textContent = result.predicted_demand;
        document.getElementById("res-safety").textContent = "+" + result.safety_stock;
        document.getElementById("res-recommended").textContent = result.recommended_preparation;
        document.getElementById("res-saved").textContent = result.meals_saved;
        document.getElementById("res-leftover").textContent = result.expected_leftover_recommended;
        document.getElementById("res-directive").textContent = 
          `Prepare approximately ${result.recommended_preparation} portions of ${payload.Food_Item} tomorrow ` +
          `(including ${result.safety_stock} portions as a 10% safety buffer). ` +
          `Compared with routine preparation of ${payload.Usual_Preparation} portions, SmartCanteen reduces food waste by ${result.meals_saved} meals (${result.waste_reduction_pct}% waste reduction)!`;

        // Render Stacked Comparison Chart
        const compTrace1 = {
          x: ['Usual Prep', 'SmartCanteen ML'],
          y: [result.predicted_demand, result.predicted_demand],
          name: 'Portions Sold',
          type: 'bar',
          marker: { color: '#2e7d32' }
        };
        const compTrace2 = {
          x: ['Usual Prep', 'SmartCanteen ML'],
          y: [result.expected_leftover_usual, result.expected_leftover_recommended],
          name: 'Leftover Surplus',
          type: 'bar',
          marker: { color: '#e53935' }
        };
        Plotly.newPlot('chart-pred-comparison', [compTrace1, compTrace2], {
          barmode: 'stack',
          margin: { t: 20, r: 20, l: 45, b: 40 },
          legend: { orientation: 'h', y: 1.15 }
        }, { responsive: true, displayModeBar: false });

        loadHistoryTable();
      }
    } catch (err) {
      alert("Prediction error: " + err);
      btn.textContent = "🔮 Forecast Demand & Kitchen Recommendation";
      btn.disabled = false;
    }
  });
}

// Waste Simulator Logic
function setupSimulator() {
  const simBtn = document.getElementById("btn-run-sim");
  if (!simBtn) return;

  simBtn.addEventListener("click", () => {
    const usual = parseInt(document.getElementById("sim-usual").value) || 0;
    const demand = parseInt(document.getElementById("sim-demand").value) || 0;
    const cost = parseFloat(document.getElementById("sim-cost").value) || 0;

    const safety = Math.round(demand * 0.10);
    const recommended = demand + safety;
    const leftoverUsual = Math.max(0, usual - demand);
    const leftoverRec = Math.max(0, recommended - demand);
    const saved = Math.max(0, usual - recommended);
    const wastePct = usual > 0 ? ((leftoverUsual / usual) * 100).toFixed(1) : 0;
    const moneyWasted = (leftoverUsual * cost).toFixed(2);
    const moneySaved = (saved * cost).toFixed(2);

    const badge = document.getElementById("sim-status");
    if (wastePct > 20) {
      badge.textContent = "High Waste (" + wastePct + "%)";
      badge.className = "kpi-value badge-red";
    } else if (wastePct > 10) {
      badge.textContent = "Moderate Waste (" + wastePct + "%)";
      badge.className = "kpi-value badge-yellow";
    } else {
      badge.textContent = "Low Waste (" + wastePct + "%)";
      badge.className = "kpi-value badge-green";
    }

    document.getElementById("sim-money-wasted").textContent = "₹" + Number(moneyWasted).toLocaleString();
    document.getElementById("sim-money-saved").textContent = "₹" + Number(moneySaved).toLocaleString();
    document.getElementById("sim-recommended").textContent = recommended + " portions";
    document.getElementById("sim-leftover-ml").textContent = leftoverRec + " portions";
  });
}

// Model Training Trigger
function setupTraining() {
  const trainBtn = document.getElementById("btn-trigger-train");
  if (!trainBtn) return;

  trainBtn.addEventListener("click", async () => {
    trainBtn.textContent = "Training 4 Models (Please Wait)...";
    trainBtn.disabled = true;

    try {
      const res = await fetch("/api/train", { method: "POST" });
      const data = await res.json();
      trainBtn.textContent = "🚀 Retrain & Benchmark Models";
      trainBtn.disabled = false;

      if (data.success) {
        alert("Training successfully completed! Champion model: " + data.best_model_name);
        loadModelInfo();
      }
    } catch (err) {
      alert("Training error: " + err);
      trainBtn.textContent = "🚀 Retrain & Benchmark Models";
      trainBtn.disabled = false;
    }
  });
}

// Upload CSV & Reset Handlers
function setupUploadAndReset() {
  const fileInput = document.getElementById("csv-file-input");
  const uploadBtn = document.getElementById("btn-upload-csv");
  const resetBtn = document.getElementById("btn-reset-dataset");

  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", async () => {
      if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please select a CSV file first.");
        return;
      }
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      uploadBtn.textContent = "Uploading & Validating...";
      uploadBtn.disabled = true;

      try {
        const res = await fetch("/api/upload-csv", { method: "POST", body: formData });
        const data = await res.json();
        uploadBtn.textContent = "Upload CSV";
        uploadBtn.disabled = false;

        if (data.success) {
          alert(data.message);
          loadDatasetInfo();
          loadAnalytics();
        } else {
          alert("Error: " + data.error);
        }
      } catch (e) {
        alert("Upload error: " + e);
        uploadBtn.textContent = "Upload CSV";
        uploadBtn.disabled = false;
      }
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      if (!confirm("Regenerate synthetic 365-day dataset?")) return;
      resetBtn.textContent = "Generating 365 Days...";
      resetBtn.disabled = true;

      try {
        const res = await fetch("/api/reset-dataset", { method: "POST" });
        const data = await res.json();
        resetBtn.textContent = "🔄 Reset / Generate Fresh 365-Day Dataset";
        resetBtn.disabled = false;
        alert(data.message);
        loadDatasetInfo();
        loadAnalytics();
      } catch (e) {
        alert("Error resetting dataset: " + e);
        resetBtn.textContent = "🔄 Reset / Generate Fresh 365-Day Dataset";
        resetBtn.disabled = false;
      }
    });
  }
}

// Load Audit History Table
async function loadHistoryTable() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    const tableBody = document.getElementById("history-table-body");
    if (!tableBody || !data.records) return;

    tableBody.innerHTML = "";
    data.records.slice().reverse().forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.Timestamp || '-'}</td>
        <td>${r.Date}</td>
        <td><b>${r.Food_Item}</b></td>
        <td>${r.Meal_Type}</td>
        <td>${r.Predicted_Demand}</td>
        <td><b>${r.Recommended_Preparation}</b></td>
        <td>${r.Usual_Preparation}</td>
        <td><span class="badge-green">+${r.Meals_Saved} portions</span></td>
        <td>${r.Status}</td>
      `;
      tableBody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading history:", err);
  }
}
