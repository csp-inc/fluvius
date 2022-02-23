FROM ghcr.io/csp-inc/fluvius:data

# Copies the code to the filesystem path `/` of the container
COPY . /content

# Code file to execute when the docker container starts up
ENTRYPOINT ["/content/src/make-docs.sh"]
