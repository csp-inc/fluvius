FROM ghcr.io/csp-inc/fluvius:data

# Copies the code to the filesystem path `/` of the container
COPY content /content
COPY src/sphinx-build.sh /src/sphinx-build.sh
# COPY . .

# Code file to execute when the docker container starts up
ENTRYPOINT ["/src/sphinx-build.sh"]
