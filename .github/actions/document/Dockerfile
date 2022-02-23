FROM ghcr.io/csp-inc/fluvius:data

# Copies the code to the filesystem path `/` of the container
COPY . .
# COPY src/sphinx-build.sh /src/sphinx-build.sh


# # NOT working
# COPY ../../../. /content
# COPY content /content

# Code file to execute when the docker container starts up
ENTRYPOINT ["/src/sphinx-build.sh"]
