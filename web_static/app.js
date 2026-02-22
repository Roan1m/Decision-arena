const statusLine = document.getElementById("statusLine");
const walletBtn = document.getElementById("walletBtn");
const walletBadge = document.getElementById("walletBadge");
const walletInfo = document.getElementById("walletInfo");
const walletGate = document.getElementById("walletGate");
const walletOverlayBtn = document.getElementById("walletOverlayBtn");

const decisionInput = document.getElementById("decisionInput");
const contextInput = document.getElementById("contextInput");
const riskInput = document.getElementById("riskInput");
const difficultyInput = document.getElementById("difficultyInput");
const horizonInput = document.getElementById("horizonInput");
const modelInput = document.getElementById("modelInput");
const runBtn = document.getElementById("runBtn");

const verdictHeadline = document.getElementById("verdictHeadline");
const confidenceScore = document.getElementById("confidenceScore");
const confidenceFill = document.getElementById("confidenceFill");
const pointViewBox = document.getElementById("pointViewBox");
const whyViewBox = document.getElementById("whyViewBox");
const concernsBox = document.getElementById("concernsBox");
const oppositeBox = document.getElementById("oppositeBox");

const frameBox = document.getElementById("frameBox");
const challengeBox = document.getElementById("challengeBox");
const briefBox = document.getElementById("briefBox");
const verdictBox = document.getElementById("verdictBox");

const ideaChips = Array.from(document.querySelectorAll(".idea-chip"));

const BASE_SEPOLIA_CHAIN_ID = "0x14a34";
const BASE_SEPOLIA_PARAMS = {
  chainId: BASE_SEPOLIA_CHAIN_ID,
  chainName: "Base Sepolia",
  nativeCurrency: { name: "Ether", symbol: "ETH", decimals: 18 },
  rpcUrls: ["https://sepolia.base.org"],
  blockExplorerUrls: ["https://sepolia.basescan.org"],
};

const state = {
  busy: false,
  walletAddress: "",
  walletChainId: "",
  feeRequired: true,
  feeAmountOpg: "0.0001",
  feeAmountWei: 100000000000000n,
  runsPerFeeTx: 10,
  remainingRuns: 0,
  feeToken: "",
  feeReceiver: "",
  feeChainId: BASE_SEPOLIA_CHAIN_ID,
};

function shortenAddress(addr) {
  if (!addr || addr.length < 10) return addr || "";
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`;
}

function setStatus(text, kind = "ok") {
  statusLine.textContent = text;
  statusLine.dataset.state = kind;
}

function isWalletReady() {
  return Boolean(state.walletAddress) && state.walletChainId === BASE_SEPOLIA_CHAIN_ID;
}

function setBusy(busy) {
  state.busy = busy;
  runBtn.disabled = busy || !isWalletReady();
  runBtn.textContent = busy ? "Running..." : isWalletReady() ? "Run Analysis" : "Connect Wallet First";
}

function syncRemainingRuns(raw) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return;
  state.remainingRuns = Math.max(0, Math.floor(n));
}

function refreshGate() {
  const locked = !isWalletReady();
  walletGate.classList.toggle("hidden", !locked);
  if (!state.busy) setBusy(false);
}

function updateWalletUi() {
  if (!state.walletAddress) {
    walletBadge.textContent = "Disconnected";
    walletBadge.dataset.mode = "neutral";
    walletBtn.textContent = "Connect Wallet";
    walletInfo.textContent = "Base Sepolia required";
    walletInfo.dataset.mode = "ok";
    refreshGate();
    return;
  }

  walletBtn.textContent = "Disconnect";
  if (state.walletChainId !== BASE_SEPOLIA_CHAIN_ID) {
    walletBadge.textContent = "Wrong Network";
    walletBadge.dataset.mode = "error";
    walletInfo.textContent = `Current chain: ${state.walletChainId || "unknown"}. Switch to Base Sepolia.`;
    walletInfo.dataset.mode = "error";
    refreshGate();
    return;
  }

  walletBadge.textContent = shortenAddress(state.walletAddress);
  walletBadge.dataset.mode = "connected";
  walletInfo.textContent = "Connected on Base Sepolia";
  walletInfo.dataset.mode = "ok";
  refreshGate();
}

function applyWalletState(address, chainId) {
  const prev = (state.walletAddress || "").toLowerCase();
  state.walletAddress = address || "";
  state.walletChainId = chainId || "";

  if (!state.walletAddress || prev !== state.walletAddress.toLowerCase()) {
    state.remainingRuns = 0;
  }

  updateWalletUi();
}

async function refreshCreditsFromServer() {
  if (!state.walletAddress) {
    state.remainingRuns = 0;
    updateWalletUi();
    return;
  }

  try {
    const res = await fetch(`/api/credits?wallet=${encodeURIComponent(state.walletAddress)}`);
    const data = await res.json();
    if (res.ok && data.ok) {
      syncRemainingRuns(data.remaining_runs);
      updateWalletUi();
    }
  } catch (_) {
    // Ignore transient credits fetch errors.
  }
}

function parseAmountToWei(amountText, decimals = 18) {
  const raw = String(amountText || "0").trim();
  if (!/^\d+(\.\d+)?$/.test(raw)) {
    throw new Error(`Invalid fee amount: ${raw}`);
  }
  const [wholeRaw, fracRaw = ""] = raw.split(".");
  const whole = BigInt(wholeRaw || "0");
  const fracPadded = (fracRaw + "0".repeat(decimals)).slice(0, decimals);
  const frac = BigInt(fracPadded || "0");
  return whole * 10n ** BigInt(decimals) + frac;
}

function encodeErc20Transfer(toAddress, amountWei) {
  const method = "a9059cbb";
  const cleanTo = String(toAddress || "").toLowerCase().replace(/^0x/, "");
  if (!/^[0-9a-f]{40}$/.test(cleanTo)) {
    throw new Error("Invalid fee receiver address.");
  }
  const encodedTo = cleanTo.padStart(64, "0");
  const encodedAmount = BigInt(amountWei).toString(16).padStart(64, "0");
  return `0x${method}${encodedTo}${encodedAmount}`;
}

function encodeErc20BalanceOf(ownerAddress) {
  const method = "70a08231";
  const clean = String(ownerAddress || "").toLowerCase().replace(/^0x/, "");
  if (!/^[0-9a-f]{40}$/.test(clean)) {
    throw new Error("Invalid wallet address.");
  }
  return `0x${method}${clean.padStart(64, "0")}`;
}

function parseHexToBigInt(hexValue) {
  return BigInt(String(hexValue || "0x0"));
}

function formatOpgWei(weiValue) {
  const v = BigInt(weiValue);
  const whole = v / 10n ** 18n;
  const frac = (v % 10n ** 18n).toString().padStart(18, "0").slice(0, 4);
  return frac === "0000" ? `${whole}` : `${whole}.${frac}`;
}

async function fetchOpgBalanceWei() {
  const data = encodeErc20BalanceOf(state.walletAddress);
  const raw = await window.ethereum.request({
    method: "eth_call",
    params: [{ to: state.feeToken, data }, "latest"],
  });
  return parseHexToBigInt(raw);
}

async function waitForReceipt(txHash, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const receipt = await window.ethereum.request({
      method: "eth_getTransactionReceipt",
      params: [txHash],
    });
    if (receipt) {
      if (receipt.status === "0x1") return receipt;
      throw new Error("Fee transaction failed on-chain.");
    }
    await new Promise((resolve) => setTimeout(resolve, 2200));
  }
  throw new Error("Fee transaction confirmation timeout.");
}

async function payFeeAndGetTxHash() {
  if (!state.feeRequired) return "";
  if (!isWalletReady()) {
    throw new Error("Wallet must be connected on Base Sepolia.");
  }
  if (!state.feeToken || !state.feeReceiver) {
    throw new Error("Fee configuration missing on server.");
  }

  const opgBalanceWei = await fetchOpgBalanceWei();
  if (opgBalanceWei < state.feeAmountWei) {
    throw new Error(
      `Insufficient OPG balance. Need ${state.feeAmountOpg}, wallet has ${formatOpgWei(opgBalanceWei)} OPG.`
    );
  }

  const tx = {
    from: state.walletAddress,
    to: state.feeToken,
    value: "0x0",
    data: encodeErc20Transfer(state.feeReceiver, state.feeAmountWei),
  };

  setStatus(`Paying ${state.feeAmountOpg} OPG...`, "ok");
  const txHash = await window.ethereum.request({
    method: "eth_sendTransaction",
    params: [tx],
  });

  setStatus("Waiting for fee confirmation...", "ok");
  await waitForReceipt(txHash);
  return txHash;
}

async function ensureFeeForRun() {
  if (!state.feeRequired) return "";
  await refreshCreditsFromServer();
  if (state.remainingRuns > 0) {
    state.remainingRuns -= 1;
    updateWalletUi();
    setStatus("Access granted.", "ok");
    return "";
  }
  return payFeeAndGetTxHash();
}

async function ensureBaseSepolia() {
  if (!window.ethereum) throw new Error("MetaMask is not available.");

  try {
    await window.ethereum.request({
      method: "wallet_switchEthereumChain",
      params: [{ chainId: BASE_SEPOLIA_CHAIN_ID }],
    });
  } catch (err) {
    if (err && err.code === 4902) {
      await window.ethereum.request({
        method: "wallet_addEthereumChain",
        params: [BASE_SEPOLIA_PARAMS],
      });
      return;
    }
    throw new Error(`Failed to switch to Base Sepolia: ${String(err?.message || err)}`);
  }
}

async function connectWallet() {
  if (!window.ethereum) {
    throw new Error("MetaMask is required.");
  }

  await ensureBaseSepolia();
  const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
  const chainId = await window.ethereum.request({ method: "eth_chainId" });
  const account = accounts && accounts[0] ? accounts[0] : "";
  applyWalletState(account, chainId);
  await refreshCreditsFromServer();
}

function disconnectWallet() {
  applyWalletState("", "");
  setStatus("Wallet disconnected", "error");
}

async function syncWalletFromProvider() {
  if (!window.ethereum) {
    applyWalletState("", "");
    return;
  }
  const accounts = await window.ethereum.request({ method: "eth_accounts" });
  const chainId = await window.ethereum.request({ method: "eth_chainId" });
  const account = accounts && accounts[0] ? accounts[0] : "";
  applyWalletState(account, chainId);
  await refreshCreditsFromServer();
}

function renderVerdictBox(verdict) {
  if (!verdict || typeof verdict !== "object") {
    verdictBox.textContent = "No verdict generated.";
    return;
  }

  verdictBox.textContent = [
    `Verdict: ${verdict.verdict || "-"}`,
    `Confidence: ${verdict.confidence_0_100 || 0}/100`,
    "",
    `Point of View: ${verdict.point_of_view || verdict.verdict || "-"}`,
    "",
    `Why This View: ${verdict.why_this_view || verdict.why_now || "-"}`,
    "",
    `Main Concerns: ${verdict.main_concerns || verdict.why_not || "-"}`,
    "",
    `Opposite View: ${verdict.opposite_view || verdict.contingency_plan || "-"}`,
  ].join("\n");
}

function clearResultsLoading() {
  verdictHeadline.textContent = "Running analysis...";
  confidenceScore.textContent = "-- / 100";
  confidenceFill.style.width = "0%";
  pointViewBox.textContent = "Generating AI point of view...";
  whyViewBox.textContent = "Generating reasoning...";
  concernsBox.textContent = "Generating main concerns...";
  oppositeBox.textContent = "Generating opposite view...";
  briefBox.textContent = "Generating short summary...";
  frameBox.textContent = "Generating decision frame...";
  challengeBox.textContent = "Generating challenge brief...";
  verdictBox.textContent = "Generating full verdict details...";
}

function renderMainSummary(result) {
  const verdict = result.verdict || {};
  const confidenceRaw = Number(verdict.confidence_0_100 || 0);
  const confidence = Math.max(0, Math.min(100, confidenceRaw));

  verdictHeadline.textContent = verdict.verdict || verdict.point_of_view || "No clear verdict";
  confidenceScore.textContent = `${confidence} / 100`;
  confidenceFill.style.width = `${confidence}%`;

  pointViewBox.textContent = verdict.point_of_view || verdict.verdict || "-";
  whyViewBox.textContent = verdict.why_this_view || verdict.why_now || "-";
  concernsBox.textContent = verdict.main_concerns || verdict.why_not || "-";
  oppositeBox.textContent = verdict.opposite_view || verdict.contingency_plan || "-";
}

function renderAll(result) {
  briefBox.textContent = result.mission_brief || "-";
  frameBox.textContent = result.decision_frame || "-";
  challengeBox.textContent = result.challenge_brief || "-";
  renderVerdictBox(result.verdict);
  renderMainSummary(result);
}

async function runAnalysis() {
  if (state.busy) return;

  const decision = decisionInput.value.trim();
  if (!decision) {
    setStatus("Decision is required.", "error");
    return;
  }
  if (!isWalletReady()) {
    setStatus("Connect wallet on Base Sepolia first.", "error");
    return;
  }

  setBusy(true);
  clearResultsLoading();
  setStatus("Preparing analysis...", "ok");

  try {
    const feeTxHash = await ensureFeeForRun();

    const payload = {
      decision,
      context: contextInput.value.trim(),
      constraints: "",
      resources: "",
      risk_appetite: riskInput.value,
      difficulty: difficultyInput.value,
      horizon_days: Number(horizonInput.value || 45),
      model: modelInput.value.trim(),
      wallet_address: state.walletAddress,
      fee_tx_hash: feeTxHash,
    };

    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `Analysis failed (${res.status})`);
    }

    syncRemainingRuns(data.remaining_runs);
    updateWalletUi();

    renderAll(data);
    setStatus(`Done | Model: ${data.model || "n/a"}`, "ok");
  } catch (err) {
    const msg = String(err?.message || err);
    verdictHeadline.textContent = "Analysis failed";
    confidenceScore.textContent = "0 / 100";
    confidenceFill.style.width = "0%";
    pointViewBox.textContent = msg;
    whyViewBox.textContent = msg;
    concernsBox.textContent = msg;
    oppositeBox.textContent = msg;
    briefBox.textContent = msg;
    frameBox.textContent = msg;
    challengeBox.textContent = msg;
    verdictBox.textContent = msg;
    setStatus(msg, "error");
  } finally {
    setBusy(false);
  }
}

async function boot() {
  setBusy(false);

  try {
    const res = await fetch("/api/ping");
    const data = await res.json();

    if (!res.ok || !data.ok) {
      throw new Error(data.error || "Ping failed");
    }

    state.feeRequired = Boolean(data.fee_required);
    state.feeAmountOpg = String(data.fee_amount_opg || state.feeAmountOpg);
    state.runsPerFeeTx = Number(data.runs_per_fee_tx || state.runsPerFeeTx);
    state.feeToken = String(data.fee_token || "");
    state.feeReceiver = String(data.fee_receiver || "");
    state.feeChainId = String(data.fee_chain_id || BASE_SEPOLIA_CHAIN_ID);
    state.feeAmountWei = parseAmountToWei(state.feeAmountOpg, 18);

    setStatus("Ready.", "ok");
  } catch (err) {
    setStatus(`Boot failed: ${String(err?.message || err)}`, "error");
  }

  try {
    await syncWalletFromProvider();
  } catch (err) {
    setStatus(`Wallet sync issue: ${String(err?.message || err)}`, "error");
  }
}

walletBtn.addEventListener("click", async () => {
  if (state.walletAddress) {
    disconnectWallet();
    return;
  }
  try {
    await connectWallet();
    setStatus("Wallet connected.", "ok");
  } catch (err) {
    setStatus(`Wallet connection failed: ${String(err?.message || err)}`, "error");
  }
});

walletOverlayBtn.addEventListener("click", async () => {
  try {
    await connectWallet();
    setStatus("Wallet connected.", "ok");
  } catch (err) {
    setStatus(`Wallet connection failed: ${String(err?.message || err)}`, "error");
  }
});

runBtn.addEventListener("click", runAnalysis);

for (const chip of ideaChips) {
  chip.addEventListener("click", () => {
    const text = chip.dataset.template || "";
    decisionInput.value = text;
    decisionInput.focus();
  });
}

if (window.ethereum) {
  window.ethereum.on("accountsChanged", async (accounts) => {
    const account = accounts && accounts[0] ? accounts[0] : "";
    const chainId = await window.ethereum.request({ method: "eth_chainId" });
    applyWalletState(account, chainId);
    await refreshCreditsFromServer();
  });

  window.ethereum.on("chainChanged", async (chainId) => {
    const accounts = await window.ethereum.request({ method: "eth_accounts" });
    const account = accounts && accounts[0] ? accounts[0] : "";
    applyWalletState(account, chainId);
    await refreshCreditsFromServer();
  });
}

boot();
