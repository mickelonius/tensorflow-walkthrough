#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
#LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis.predict_pb2 import PredictRequest
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import json
import requests
from io import BytesIO
import gzip


def zip_payload(payload: str) -> bytes:
    btsio = BytesIO()
    g = gzip.GzipFile(fileobj=btsio, mode='w')
    g.write(bytes(payload, 'utf8'))
    g.close()
    return btsio.getvalue()

def main():
    #x_new = np.load('/tmp/my_mnist_tests.npy')
    #x_new = np.load('/home/mike/Repos/CaseStudies/Arturo/my_mnist_tests.npy')
    x_new = np.load('my_mnist_tests.npy')
    model_name = "my_mnist_model"

    request = PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = "serving_default"
    #input_name = saved_model.input_names[0]
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(x_new))

    channel = grpc.insecure_channel('localhost:8500')
    predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    response = predict_service.Predict(request, timeout=10.0)
    print(response)


    data_payload = {
        "signature_name": "serving_default",
        "instances": x_new.tolist(),
    }
    SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
    zipped_payload = zip_payload(json.dumps(data_payload))
    response = requests.post(SERVER_URL, data=zipped_payload, headers={'Content-Encoding': 'gzip'})
    #response = requests.post(SERVER_URL, data=json.dumps(data_payload))
    response.raise_for_status()
    response = response.json()
    print('HTTP Response:')
    print(json.dumps(response, indent=2))


if __name__ == '__main__':
    main()