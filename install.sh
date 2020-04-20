#!/usr/bin/env bash
# *** Please run this from project root directory ***

#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
#LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#########################################################
# Setup/Docker
#########################################################
if [[ $(which docker) && $(docker --version) ]]; then
    echo "$(docker --version)"
  else
    echo "Please install docker to run this project"
    exit
fi

docker pull tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
docker pull tensorflow/serving:latest


#########################################################
# Train ResNet model, save for use in server
#########################################################
case $1 in

    --train)
    CONTAINERNAME="arturo_train"
    docker run --name $CONTAINERNAME --gpus all --rm -p 8888:8888 \
    --mount type=bind,source=$PWD,target=/arturo \
    tensorflow/tensorflow:2.1.0-gpu-py3-jupyter &
    #-v $PWD:/arturo \


    echo "  Waiting for arturo_train container...."
    until [ "$(docker ps -a | grep ${CONTAINERNAME})" ]
    do
      sleep 2;
    done

    docker exec -ti arturo_train python /arturo/train.py
    docker stop $CONTAINERNAME
    shift # past argument=value
    ;;

    *)
    echo "Skip training"
    shift # past argument with no value
    ;;

esac


#########################################################
# Run backend server and test client
#########################################################
CONTAINERNAME="arturo_backend"
MODEL_PATH="$PWD/my_mnist_model"
docker run --name arturo_backend --rm -p 8501:8501 -p 8500:8500  \
--mount type=bind,source=$MODEL_PATH,target=/models/my_mnist_model \
-e MODEL_NAME=my_mnist_model -t tensorflow/serving &

echo "  Waiting for arturo_backend container...."
until [ "$(docker ps -a | grep ${CONTAINERNAME})" ]
do
  sleep 2;
done

case $2 in

    --venv)
    python3.6 -m venv ./venv
    source ./venv/bin/activate
    pip install -r ./requirements.txt
    ;;

    *)
    source venv/bin/activate
    shift # past argument with no value
    ;;

esac

python client.py
deactivate

docker stop arturo_backend

