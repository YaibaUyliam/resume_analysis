#–– Logging helpers ––#
info()    { echo -e "ℹ  $*"; }
success() { echo -e "✅ $*"; }
error()   { echo -e "❌ $*" >&2; exit 1; }

# ensure uv
if ! command -v uv &> /dev/null; then
  info "uv not found; installing via Astral.sh…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Ollama & model setup
ollama_check_or_pull() {
      model_name="$1"
      if ! ollama list | grep -q "$model_name"; then
	  info "Pulling $model_name model…"
	  ollama pull "$model_name" || error "Failed to pull $model_name model"
	  success "$model_name model ready"
      else
	  info "$model_name model already present—skipping"
      fi
}

info "Checking Ollama installation…"
if ! command -v ollama &> /dev/null; then
  info "ollama not found; installing…"

  if [[ "$OS_TYPE" == "macOS" ]]; then
    brew install ollama || error "Failed to install Ollama via Homebrew"
  else
    # Download Ollama installer securely without using curl | sh
    curl -Lo ollama-install.sh https://ollama.com/install.sh || error "Failed to download Ollama installer"
    chmod +x ollama-install.sh
    ./ollama-install.sh || error "Failed to execute Ollama installer"
    rm ollama-install.sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  success "Ollama installed"
fi

info "Setting up backend"
(
  info "Syncing Python deps via uv…"
  uv sync
  source .venv/bin/activate
  uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install transformers==4.53.3
  uv pip install accelerate
  uv pip install wheel
  uv pip install flash-attn --no-build-isolation
  success "Backend dependencies ready."

  # The Ollama provider automatically pulls models on demand, but it's preferable to do it at setup time.
  eval `grep ^LLM_PROVIDER= .env`
  if [ "$LLM_PROVIDER" = "ollama" ]; then
      eval `grep ^LL_MODEL .env`
      ollama_check_or_pull $LL_MODEL
  fi
  eval `grep ^EMBEDDING_PROVIDER= .env`
  if [ "$EMBEDDING_PROVIDER" = "ollama" ]; then
      eval `grep ^EMBEDDING_MODEL .env`
      ollama_check_or_pull $EMBEDDING_MODEL
  fi
)
