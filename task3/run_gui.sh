#!/bin/bash
#
# Start the Task 3 GUI.
#
#   ./run_gui.sh
#
# The server runs on tf-env's interpreter because that is where TensorFlow lives
# and three of the four models are TensorFlow.  The Task 1 forest is loaded by a
# helper process on mvi-env, which app.py starts by itself when it is needed -
# nothing has to be activated by hand.
#
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"

SERVER_PYTHON="$PROJECT/tf-env/bin/python"
WORKER_PYTHON="$PROJECT/mvi-env/bin/python"

if [ ! -x "$SERVER_PYTHON" ]; then
  echo "error: tf-env not found at $SERVER_PYTHON"
  echo "       the GUI needs the Task 2 environment to serve the .h5 models."
  exit 1
fi

if [ ! -x "$WORKER_PYTHON" ]; then
  echo "warning: mvi-env not found at $WORKER_PYTHON"
  echo "         Task 2 models will work, but the Task 1 forest will not load."
fi

if ! "$SERVER_PYTHON" -c "import flask" 2>/dev/null; then
  echo "installing Flask into tf-env ..."
  "$SERVER_PYTHON" -m pip install --quiet flask
fi

# The sound effects are generated rather than downloaded, so make them if this is
# a fresh checkout.
if [ ! -f "$HERE/static/sounds/model_loaded.wav" ]; then
  echo "generating sound effects ..."
  (cd "$HERE" && "$SERVER_PYTHON" make_sounds.py)
fi

cd "$HERE"
echo
echo "starting the GUI on http://127.0.0.1:${MVI_GUI_PORT:-5050}"
exec "$SERVER_PYTHON" app.py
