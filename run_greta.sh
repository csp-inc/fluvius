#!/usr/bin/env bash
REPO=$(basename $(pwd))
PORT=8662
ACCESS_POINT=http://localhost:$PORT/

IMAGE=csp-inc/fluvius-greta:latest
DPATH=docker/greta.Dockerfile

if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  docker build --rm -t $IMAGE - < $DPATH
fi

echo "$ACCESS_POINT (with usr and pwd $REPO)"
docker run -it --rm \
    -v "$(pwd)":/home/$REPO \
    -v "/home/$USER/.rstudio":/home/$REPO/.rstudio \
    -v "/home/$USER/themes/rstudio":/etc/rstudio/themes/ \
    -w "/home/$REPO" \
    -e USER=$REPO \
    -e PASSWORD=$REPO \
    -p $PORT:8787 \
    $IMAGE "$@"

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Change back ownership to user
    sudo chown -R $USER:$USER .
    sudo chown -R $USER:$USER /home/$USER/.rstudio
    sudo chown -R $USER:$USER /home/$USER/themes
fi

# Remove crap left over in the RStudio settings 
# so the same settings can still be used in other 
# repos
rm -rf /home/$USER/.rstudio/sources/ 
rm -rf /home/$USER/.rstudio/notebooks/
rm -rf /home/$USER/.rstudio/monitored/lists/
rm -f /home/$USER/.rstudio/history_database
# removes old session to prevent unneccessary crash error on startup
rm -rf /home/$USER/.rstudio/sessions/ 

# remove local copies created in the repo
rm -rf .rstudio/ kitematic/ rstudio/
