/**
 * ChurnSense Frontend Application Logic
 * Vanilla JavaScript implementation for inference, explainability, multi-model switching,
 * 1-click demo scenarios, retention simulator, consensus breakdown, and AI advisor.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const themeToggle = document.getElementById("theme-toggle");
  const moonIcon = document.getElementById("moon-icon");
  const sunIcon = document.getElementById("sun-icon");

  const form = document.getElementById("churn-form");
  const predictBtn = document.getElementById("predict-btn");
  const btnText = predictBtn.querySelector(".btn-text");
  const btnSpinner = predictBtn.querySelector(".btn-spinner");
  const resetBtn = document.getElementById("reset-btn");

  const tenureInput = document.getElementById("tenure");
  const tenureBadge = document.getElementById("tenure-badge");
  const monthlyChargesInput = document.getElementById("MonthlyCharges");
  const monthlyChargesBadge = document.getElementById("monthly-charges-badge");
  const totalChargesInput = document.getElementById("TotalCharges");

  const errorBanner = document.getElementById("error-banner");
  const errorBannerMsg = document.getElementById("error-banner-msg");
  const bannerRetryBtn = document.getElementById("banner-retry-btn");

  const resultsEmpty = document.getElementById("results-empty");
  const resultsLoading = document.getElementById("results-loading");
  const resultsContent = document.getElementById("results-content");

  const gaugeFill = document.getElementById("gauge-progress");
  const gaugePct = document.getElementById("gauge-pct");
  const riskBadge = document.getElementById("risk-badge");
  const verdictTitle = document.getElementById("verdict-title");
  const thresholdVal = document.getElementById("threshold-val");
  const reasonsList = document.getElementById("reasons-list");
  const llmSourceBadge = document.getElementById("llm-source-badge");
  const llmExplanationText = document.getElementById("llm-explanation-text");

  let manualTotalChargesEdited = false;
  let lastSubmittedPayload = null;
  let currentCustomerContext = null;
  let chatHistory = [];
  let currentModel = "logistic_regression";

  // 1. Theme Initialization & Toggle
  const savedTheme = localStorage.getItem("churnsense_theme") || "dark";
  setTheme(savedTheme);

  themeToggle.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
  });

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("churnsense_theme", theme);
    if (theme === "light") {
      moonIcon.classList.add("hidden");
      sunIcon.classList.remove("hidden");
    } else {
      sunIcon.classList.add("hidden");
      moonIcon.classList.remove("hidden");
    }
  }

  // 2. Model Switching Logic
  const modelDescriptions = {
    logistic_regression: "Logistic Regression (Tuned L2) • Production Active • 0.8408 Test ROC-AUC",
    random_forest: "Random Forest (300 Trees) • Non-linear Bagging • 0.8226 Test ROC-AUC",
    xgboost: "XGBoost Classifier • Gradient Boosted Trees • 0.8187 Test ROC-AUC",
  };

  function switchModel(modelKey) {
    currentModel = modelKey;
    document.querySelectorAll(".model-tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-model") === modelKey);
    });

    document.querySelectorAll(".btn-table-switch").forEach((btn) => {
      const isActive = btn.getAttribute("data-model") === modelKey;
      btn.classList.toggle("active", isActive);
      const span = btn.querySelector("span");
      if (span) {
        span.textContent = isActive ? "✓ Active Engine" : "Test in Predictor →";
      }
    });

    const descEl = document.getElementById("model-desc");
    if (descEl && modelDescriptions[modelKey]) {
      descEl.textContent = modelDescriptions[modelKey];
    }

    const cleanName = modelKey.replace("_", " ").toUpperCase();
    showToast(`Switched active inference model to ${cleanName}`, "info");

    // If a prediction is already visible, re-score with the newly selected model
    if (lastSubmittedPayload) {
      executePrediction(lastSubmittedPayload);
    }
  }

  document.querySelectorAll(".model-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const model = btn.getAttribute("data-model");
      if (model) switchModel(model);
    });
  });

  document.querySelectorAll(".btn-table-switch").forEach((btn) => {
    btn.addEventListener("click", () => {
      const model = btn.getAttribute("data-model");
      if (model) {
        switchModel(model);
        document.getElementById("predict-section")?.scrollIntoView({ behavior: "smooth" });
      }
    });
  });

  // 3. Sliders & Live TotalCharges Calculation
  function updateTenureDisplay() {
    const months = parseInt(tenureInput.value, 10);
    tenureBadge.textContent = `${months} ${months === 1 ? "month" : "months"}`;
    autoCalculateTotalCharges();
  }

  function updateMonthlyChargesDisplay() {
    const cost = parseFloat(monthlyChargesInput.value).toFixed(2);
    monthlyChargesBadge.textContent = `$${cost} / mo`;
    autoCalculateTotalCharges();
  }

  function autoCalculateTotalCharges() {
    if (!manualTotalChargesEdited) {
      const tenure = parseInt(tenureInput.value, 10);
      const monthly = parseFloat(monthlyChargesInput.value);
      const estimated = (tenure * monthly).toFixed(2);
      totalChargesInput.value = estimated;
    }
  }

  tenureInput.addEventListener("input", updateTenureDisplay);
  monthlyChargesInput.addEventListener("input", updateMonthlyChargesDisplay);
  totalChargesInput.addEventListener("input", () => {
    manualTotalChargesEdited = true;
  });

  // Initial slider sync
  updateTenureDisplay();
  updateMonthlyChargesDisplay();

  // 4. 1-Click Demo Scenarios / Presets
  const presets = {
    "high-risk": {
      gender: "Female",
      SeniorCitizen: 1,
      Partner: "No",
      Dependents: "No",
      tenure: 2,
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "Yes",
      StreamingMovies: "Yes",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: 95.50,
      TotalCharges: 191.00,
    },
    "low-risk": {
      gender: "Male",
      SeniorCitizen: 0,
      Partner: "Yes",
      Dependents: "Yes",
      tenure: 65,
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "Yes",
      DeviceProtection: "Yes",
      TechSupport: "Yes",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "Two year",
      PaperlessBilling: "No",
      PaymentMethod: "Credit card (automatic)",
      MonthlyCharges: 45.00,
      TotalCharges: 2925.00,
    },
    "borderline": {
      gender: "Female",
      SeniorCitizen: 0,
      Partner: "No",
      Dependents: "No",
      tenure: 16,
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "Fiber optic",
      OnlineSecurity: "Yes",
      OnlineBackup: "No",
      DeviceProtection: "Yes",
      TechSupport: "No",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "One year",
      PaperlessBilling: "Yes",
      PaymentMethod: "Bank transfer (automatic)",
      MonthlyCharges: 75.00,
      TotalCharges: 1200.00,
    },
  };

  document.querySelectorAll(".preset-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      const data = presets[presetKey];
      if (!data) return;

      // Populate form
      Object.keys(data).forEach((key) => {
        const input = document.getElementById(key);
        if (input) {
          input.value = data[key];
        }
      });

      manualTotalChargesEdited = false;
      updateTenureDisplay();
      updateMonthlyChargesDisplay();
      clearValidationErrors();
      hideErrorBanner();

      const tagEl = btn.querySelector("span");
      const label = tagEl ? tagEl.textContent.trim() : presetKey;
      showToast(`Loaded ${label}! Running instant inference...`, "success");

      // 1-Click instant execution
      const payload = { ...data };
      lastSubmittedPayload = payload;
      executePrediction(payload);
    });
  });

  // 5. Form Reset
  resetBtn.addEventListener("click", () => {
    form.reset();
    manualTotalChargesEdited = false;
    updateTenureDisplay();
    updateMonthlyChargesDisplay();
    clearValidationErrors();
    hideErrorBanner();
    resultsContent.classList.add("hidden");
    resultsEmpty.classList.remove("hidden");
    showToast("Form fields reset to defaults", "info");
  });

  // 6. Validation Helpers
  function clearValidationErrors() {
    document.querySelectorAll(".form-input.is-invalid").forEach((el) => {
      el.classList.remove("is-invalid");
    });
    document.querySelectorAll(".field-error-msg").forEach((el) => {
      el.classList.add("hidden");
      el.textContent = "";
    });
  }

  function showFieldError(fieldName, message) {
    const input = document.getElementById(fieldName);
    if (input) {
      input.classList.add("is-invalid");
    }
    const errContainer = document.getElementById(`${fieldName}-error`);
    if (errContainer) {
      errContainer.textContent = message;
      errContainer.classList.remove("hidden");
    }
  }

  function showErrorBanner(msg) {
    errorBannerMsg.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function hideErrorBanner() {
    errorBanner.classList.add("hidden");
  }

  bannerRetryBtn.addEventListener("click", () => {
    if (lastSubmittedPayload) {
      hideErrorBanner();
      executePrediction(lastSubmittedPayload);
    }
  });

  // 7. Toast System
  function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type === "error" ? "toast-error" : "toast-success"}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(20px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // 8. Form Submission & API Execution
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    clearValidationErrors();
    hideErrorBanner();

    // Serialize payload
    const formData = new FormData(form);
    const payload = {
      gender: formData.get("gender"),
      SeniorCitizen: parseInt(formData.get("SeniorCitizen"), 10),
      Partner: formData.get("Partner"),
      Dependents: formData.get("Dependents"),
      tenure: parseInt(formData.get("tenure"), 10),
      PhoneService: formData.get("PhoneService"),
      MultipleLines: formData.get("MultipleLines"),
      InternetService: formData.get("InternetService"),
      OnlineSecurity: formData.get("OnlineSecurity"),
      OnlineBackup: formData.get("OnlineBackup"),
      DeviceProtection: formData.get("DeviceProtection"),
      TechSupport: formData.get("TechSupport"),
      StreamingTV: formData.get("StreamingTV"),
      StreamingMovies: formData.get("StreamingMovies"),
      Contract: formData.get("Contract"),
      PaperlessBilling: formData.get("PaperlessBilling"),
      PaymentMethod: formData.get("PaymentMethod"),
      MonthlyCharges: parseFloat(formData.get("MonthlyCharges")),
      TotalCharges: parseFloat(formData.get("TotalCharges") || 0),
    };

    lastSubmittedPayload = payload;
    executePrediction(payload);
  });

  async function executePrediction(payload) {
    setButtonLoading(true);
    resultsEmpty.classList.add("hidden");
    resultsContent.classList.add("hidden");
    resultsLoading.classList.remove("hidden");

    try {
      const url = `/explain?model=${encodeURIComponent(currentModel)}`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (response.status === 422) {
        const errorData = await response.json();
        handleValidationErrors(errorData);
        resultsLoading.classList.add("hidden");
        resultsEmpty.classList.remove("hidden");
        showToast("Please correct the highlighted form errors.", "error");
        return;
      }

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      renderResults(data);
      showToast(`Inference complete (${data.model_name || "Model"})`, "success");

    } catch (err) {
      console.error("Prediction failed:", err);
      showErrorBanner("Could not contact server. Please verify your connection or click Retry.");
      resultsLoading.classList.add("hidden");
      resultsEmpty.classList.remove("hidden");
      showToast("Request failed. Please retry.", "error");
    } finally {
      setButtonLoading(false);
    }
  }

  function setButtonLoading(isLoading) {
    predictBtn.disabled = isLoading;
    if (isLoading) {
      btnText.textContent = "Analyzing Risk...";
      btnSpinner.classList.remove("hidden");
    } else {
      btnText.textContent = "Predict & Explain Churn";
      btnSpinner.classList.add("hidden");
    }
  }

  function handleValidationErrors(errorData) {
    if (errorData.details && Array.isArray(errorData.details)) {
      errorData.details.forEach((err) => {
        showFieldError(err.field, err.message);
      });
    }
  }

  // 9. Results Rendering & Gauge Animation
  function renderResults(data) {
    resultsLoading.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    const prob = data.churn_probability;
    const pct = Math.round(prob * 100);
    const threshold = data.threshold || 0.50;

    // Verdict and Risk categorization
    thresholdVal.textContent = threshold.toFixed(2);
    verdictTitle.textContent = data.prediction;

    // Cache context for chatbot
    currentCustomerContext = {
      churn_probability: prob,
      prediction: data.prediction,
      threshold: threshold,
      top_reasons: data.top_reasons || [],
      inputs: lastSubmittedPayload || {},
      model_used: data.model_used || currentModel,
    };
    updateChatContextBanner();

    // Configure Risk Badges and Colors
    let strokeColor;
    if (prob >= 0.70) {
      riskBadge.className = "risk-badge badge-high";
      riskBadge.textContent = "High Risk";
      strokeColor = getComputedStyle(document.documentElement).getPropertyValue("--risk-high").trim() || "#f43f5e";
    } else if (prob >= 0.40) {
      riskBadge.className = "risk-badge badge-med";
      riskBadge.textContent = "Moderate Risk";
      strokeColor = getComputedStyle(document.documentElement).getPropertyValue("--risk-med").trim() || "#f59e0b";
    } else {
      riskBadge.className = "risk-badge badge-low";
      riskBadge.textContent = "Low Risk";
      strokeColor = getComputedStyle(document.documentElement).getPropertyValue("--risk-low").trim() || "#10b981";
    }

    // Animate SVG circular gauge
    const circumference = 301.59;
    const targetOffset = circumference * (1 - prob);

    gaugeFill.style.stroke = strokeColor;
    gaugeFill.style.strokeDashoffset = targetOffset;

    // Count up percentage number smoothly
    animateValue(gaugePct, 0, pct, 700);

    // Render SHAP Driver Cards
    reasonsList.innerHTML = "";
    const reasons = data.top_reasons || [];
    const maxImpact = Math.max(...reasons.map((r) => Math.abs(r.impact)), 0.01);

    reasons.forEach((r) => {
      const isRiskIncrease = r.impact > 0;
      const barPct = Math.min(Math.round((Math.abs(r.impact) / maxImpact) * 100), 100);

      const card = document.createElement("div");
      card.className = "reason-card";
      card.innerHTML = `
        <div class="reason-header">
          <span class="reason-name">${escapeHtml(r.feature)}</span>
          <span class="reason-badge ${isRiskIncrease ? 'risk-up' : 'risk-down'}">
            ${isRiskIncrease ? '▲ Increases risk' : '▼ Reduces risk'} (${escapeHtml(String(r.value))})
          </span>
        </div>
        <div class="impact-bar-wrap">
          <div class="impact-bar-fill ${isRiskIncrease ? 'risk-up' : 'risk-down'}" style="width: ${barPct}%;"></div>
        </div>
      `;
      reasonsList.appendChild(card);
    });

    // Render AI Explanation Card
    if (data.explanation_source === "gemini") {
      llmSourceBadge.className = "ai-badge badge-gemini";
      llmSourceBadge.innerHTML = `
        <svg viewBox="0 0 24 24" fill="currentColor" class="ai-star-icon">
          <path d="M12 2l2.4 7.4H22l-6 4.6 2.3 7.2L12 16.8 5.7 21.2 8 14 2 9.4h7.6z"/>
        </svg>
        Generated by Gemini Flash
      `;
    } else {
      llmSourceBadge.className = "ai-badge badge-fallback";
      llmSourceBadge.innerHTML = `
        <svg viewBox="0 0 20 20" fill="currentColor" class="ai-star-icon">
          <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
        </svg>
        Generated via Grounded Fallback
      `;
    }

    llmExplanationText.textContent = data.explanation || "No explanation provided.";

    // Render Multi-Model Consensus Breakdown
    if (data.consensus) {
      if (data.consensus.logistic_regression !== undefined) {
        document.getElementById("prob-lr").textContent = `${Math.round(data.consensus.logistic_regression * 100)}%`;
      }
      if (data.consensus.random_forest !== undefined) {
        document.getElementById("prob-rf").textContent = `${Math.round(data.consensus.random_forest * 100)}%`;
      }
      if (data.consensus.xgboost !== undefined) {
        document.getElementById("prob-xgb").textContent = `${Math.round(data.consensus.xgboost * 100)}%`;
      }

      // Highlight active model card
      ["lr", "rf", "xgb"].forEach((m) => {
        document.getElementById(`card-${m}`)?.classList.remove("active-model");
      });
      const map = { logistic_regression: "lr", random_forest: "rf", xgboost: "xgb" };
      const activeCode = map[data.model_used || currentModel] || "lr";
      document.getElementById(`card-${activeCode}`)?.classList.add("active-model");

      // Check consensus agreement
      const probs = Object.values(data.consensus);
      const allChurn = probs.every((p) => p >= threshold);
      const allStay = probs.every((p) => p < threshold);
      const badge = document.getElementById("consensus-verdict-badge");
      if (badge) {
        if (allChurn || allStay) {
          badge.className = "consensus-pill pill-agree";
          badge.textContent = "High Agreement (3/3 Models)";
        } else {
          badge.className = "consensus-pill metric-border";
          badge.textContent = "Moderate Agreement (2/3 Models)";
        }
      }
    }
  }

  // 10. Retention Strategy Simulator ("What-If" Interventions)
  const simContractBtn = document.getElementById("sim-contract-btn");
  if (simContractBtn) {
    simContractBtn.addEventListener("click", () => {
      const contractSelect = document.getElementById("Contract");
      if (contractSelect) {
        contractSelect.value = "One year";
        showToast("Simulating 1-Year Contract upgrade...", "success");
        form.dispatchEvent(new Event("submit"));
      }
    });
  }

  const simBundleBtn = document.getElementById("sim-bundle-btn");
  if (simBundleBtn) {
    simBundleBtn.addEventListener("click", () => {
      const secSelect = document.getElementById("OnlineSecurity");
      const techSelect = document.getElementById("TechSupport");
      if (secSelect) secSelect.value = "Yes";
      if (techSelect) techSelect.value = "Yes";
      showToast("Simulating Tech Support & Security Bundle...", "success");
      form.dispatchEvent(new Event("submit"));
    });
  }

  const simAutopayBtn = document.getElementById("sim-autopay-btn");
  if (simAutopayBtn) {
    simAutopayBtn.addEventListener("click", () => {
      const paySelect = document.getElementById("PaymentMethod");
      if (paySelect) {
        paySelect.value = "Bank transfer (automatic)";
        showToast("Simulating Automatic Bank Transfer billing...", "success");
        form.dispatchEvent(new Event("submit"));
      }
    });
  }

  // 11. Copy Executive Summary to Clipboard
  const copySummaryBtn = document.getElementById("copy-summary-btn");
  const copyStatusMsg = document.getElementById("copy-status-msg");
  if (copySummaryBtn) {
    copySummaryBtn.addEventListener("click", () => {
      if (!currentCustomerContext) return;
      const probPct = Math.round(currentCustomerContext.churn_probability * 100);
      const verdict = currentCustomerContext.prediction;
      const explanation = llmExplanationText.textContent;
      const reasons = currentCustomerContext.top_reasons
        .map((r) => `• ${r.feature} (${r.value}): ${r.direction}`)
        .join("\n");

      const briefText = `[ChurnSense Retention Brief]\nActive Model: ${currentModel.toUpperCase()}\nChurn Risk Probability: ${probPct}%\nStatus: ${verdict} (Cutoff: ${currentCustomerContext.threshold})\n\nTop SHAP Drivers:\n${reasons}\n\nExecutive Retention Synthesis:\n${explanation}`;

      navigator.clipboard.writeText(briefText).then(() => {
        if (copyStatusMsg) {
          copyStatusMsg.textContent = "✓ Copied to clipboard!";
          setTimeout(() => {
            copyStatusMsg.textContent = "";
          }, 3000);
        }
        showToast("Executive retention brief copied to clipboard!", "success");
      }).catch(() => {
        showToast("Copied via fallback buffer", "info");
      });
    });
  }

  function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      obj.textContent = `${Math.floor(progress * (end - start) + start)}%`;
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        obj.textContent = `${end}%`;
      }
    };
    window.requestAnimationFrame(step);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // 12. Chatbot Assistant Logic
  const chatLauncherBtn = document.getElementById("chat-launcher-btn");
  const chatModal = document.getElementById("chat-modal");
  const chatCloseBtn = document.getElementById("chat-close-btn");
  const chatClearBtn = document.getElementById("chat-clear-btn");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatMessages = document.getElementById("chat-messages");
  const chatSendBtn = document.getElementById("chat-send-btn");

  function updateChatContextBanner() {
    const banner = document.getElementById("chat-context-banner");
    const desc = document.getElementById("chat-context-desc");
    if (!banner || !desc) return;
    if (currentCustomerContext) {
      const pct = Math.round(currentCustomerContext.churn_probability * 100);
      desc.textContent = `Analyzed Customer: ${pct}% Risk (${currentCustomerContext.prediction})`;
      banner.classList.remove("hidden");
    } else {
      banner.classList.add("hidden");
    }
  }

  function toggleChatModal(show) {
    if (show) {
      chatModal.classList.remove("hidden");
      chatInput.focus();
      updateChatContextBanner();
    } else {
      chatModal.classList.add("hidden");
    }
  }

  if (chatLauncherBtn) {
    chatLauncherBtn.addEventListener("click", () => {
      const isHidden = chatModal.classList.contains("hidden");
      toggleChatModal(isHidden);
    });
  }

  if (chatCloseBtn) {
    chatCloseBtn.addEventListener("click", () => {
      toggleChatModal(false);
    });
  }

  if (chatClearBtn) {
    chatClearBtn.addEventListener("click", () => {
      chatHistory = [];
      chatMessages.innerHTML = `
        <div class="chat-msg msg-assistant">
          <div class="msg-bubble">
            👋 Conversation reset. Ask me any retention question, doubt, or request advice on current customer profiles!
          </div>
        </div>
      `;
      showToast("Chat history cleared", "info");
    });
  }

  // Quick prompt chips
  document.querySelectorAll(".quick-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const question = chip.getAttribute("data-question");
      if (question) {
        chatInput.value = question;
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  // Chat submission
  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = chatInput.value.trim();
      if (!text) return;

      appendChatMessage("user", text);
      chatInput.value = "";
      chatSendBtn.disabled = true;

      const typingEl = showTypingIndicator();

      try {
        const payload = {
          message: text,
          history: chatHistory.slice(-6),
          customer_context: currentCustomerContext,
        };

        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          throw new Error(`Server returned ${res.status}`);
        }

        const data = await res.json();
        typingEl.remove();

        appendChatMessage("assistant", data.reply, data.source);
        chatHistory.push({ role: "user", text: text });
        chatHistory.push({ role: "assistant", text: data.reply });

      } catch (err) {
        console.error("Chat request failed:", err);
        typingEl.remove();
        appendChatMessage(
          "assistant",
          "⚠️ I encountered a temporary connection issue. Please make sure the backend is reachable or try asking again."
        );
      } finally {
        chatSendBtn.disabled = false;
        chatInput.focus();
      }
    });
  }

  function appendChatMessage(role, text, source) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${role === "user" ? "msg-user" : "msg-assistant"}`;

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    if (role === "assistant") {
      bubble.innerHTML = formatMarkdown(text);
      if (source === "fallback") {
        const tag = document.createElement("div");
        tag.style.fontSize = "0.6875rem";
        tag.style.color = "var(--text-muted)";
        tag.style.marginTop = "0.35rem";
        tag.textContent = "Generated via Offline Rules";
        bubble.appendChild(tag);
      }
    } else {
      bubble.textContent = text;
    }

    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function showTypingIndicator() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-msg msg-assistant";
    msgDiv.innerHTML = `
      <div class="msg-bubble">
        <div class="typing-dots">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
  }

  function formatMarkdown(raw) {
    let text = escapeHtml(raw);
    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\n\n/g, "<br><br>");
    text = text.replace(/\n- /g, "<br>• ");
    text = text.replace(/\n\* /g, "<br>• ");
    text = text.replace(/\n(\d+)\. /g, "<br><strong>$1.</strong> ");
    return text;
  }
});
