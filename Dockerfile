# Headless boot environment for the settled-computer MCP server.
#
# Used for registry checks (e.g. Glama) that start the server in a container and
# send MCP introspection requests. Xvfb provides a virtual display so pyautogui
# can import; the server then answers initialize/tools/list normally.
# Real screen capture and input injection require a real desktop session.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends xvfb xauth tk scrot \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "settled-computer[desktop]"

ENV DISPLAY=:99

# mss insists on an Xauthority file even for an unauthenticated Xvfb display.
CMD ["sh", "-c", "touch /root/.Xauthority; Xvfb :99 -screen 0 1280x720x24 -nolisten tcp & sleep 1; exec settled-computer"]
