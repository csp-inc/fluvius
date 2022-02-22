FROM ghcr.io/csp-inc/fluvius:data

# Copies the code to the filesystem path `/` of the container
COPY . .

# Code file to execute when the docker container starts up
ENTRYPOINT ["/src/make-docs.sh"]
