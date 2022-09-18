import timeit
import sys
import os
print(sys.path)
pth = os.path.abspath("./src/py")
print(f"add to path: {pth}")
sys.path.append(pth)
#r"C:\Users\MSI-NB\Source\Repos\FANTOME-PAN\flower\src\py"
from flwr_crypto_cpp import create_shares, combine_shares
from flwr.common.light_sec_agg import light_sec_agg_test
from flwr.common.sec_agg_plus import sec_agg_test
import random

#from flwr.common.sec_agg.sec_agg_primitives import combine_shares, create_shares
'''weights: Weights = [np.array([[-0.2, -0.5, 1.9], [0.0, 2.4, -1.9]]),
                    np.array([[0.2, 0.5, -1.9], [0.0, -2.4, 1.9]])]
quantized_weights = sec_agg_primitives.quantize(
    weights, 3, 10)
quantized_weights = sec_agg_primitives.weights_divide(quantized_weights, 4)
print(quantized_weights)'''


def test_combine_shares() -> None:
    x = timeit.default_timer()
    message = b"Quack quack!"
    share_num = 1400
    threshold = 700
    shares = create_shares(message, threshold, share_num)
    shares_collected = random.sample(shares, threshold)
    message_constructed = combine_shares(shares_collected)
    assert(message == message_constructed)
    y = timeit.default_timer()
    print(y-x)


if __name__ == "__main__":
    
    #sec_agg_primitives_test.test_all()
    #test_combine_shares()
    f = open("log.txt", "w")
    f.write("Starting real experiments\n")
    f.close()
    f = open("logserver.txt", "w")
    f.write("Starting real experiments\n")
    f.close()
    f = open("logclient.txt", "w")
    f.write("Starting real experiments\n")
    f.close()
    # sample_num_list = [100, 200, 300, 400, 500]
    sample_num_list = [10]
    dropout_value_list = [1]
    # dropout_value_list = [0]
    # vector_dimension_list = [100000, 200000, 300000, 400000, 500000]
    vector_dimension_list = [100000]
    # for vector_dimension in vector_dimension_list:
    #     for sample_num in [50]:
    #         for dropout_value in dropout_value_list:
    #             for i in range(1):
    #                 f = open("log.txt", "a")
    #                 f.write(
    #                     f"This is secagg sampling {sample_num} dropping out {dropout_value*5}% with vector size {vector_dimension} try {i} \n")
    #                 f.close()
    #                 f = open("logserver.txt", "a")
    #                 f.write(
    #                     f"This is secagg sampling {sample_num} dropping out {dropout_value*5}% with vector size {vector_dimension} try {i} \n")
    #                 f.close()
    #                 f = open("logclient.txt", "a")
    #                 f.write(
    #                     f"This is secagg sampling {sample_num} dropping out {dropout_value*5}% with vector size {vector_dimension} try {i} \n")
    #                 f.close()
    #                 sec_agg_test.test_start_simulation(
    #                     sample_num=sample_num, share_num=7, threshold=4, vector_dimension=vector_dimension, dropout_value=dropout_value, num_rounds=1)
    #                 # light_sec_agg_test.test_start_simulation(
    #                 #     sample_num=sample_num,
    #                 #     T=4, U=7, vector_dimension=vector_dimension, dropout_value=dropout_value
    #                 # )

    for vector_dimension in [100000]:
        for sample_num in sample_num_list:
            for dropout_value in dropout_value_list:
                for i in range(1):
                    f = open("log.txt", "a")
                    f.write(
                        f"This is secagg sampling {sample_num} dropping out {dropout_value*5}% with vector size {vector_dimension} try {i} \n")
                    f.close()
                    f = open("logserver.txt", "a")
                    f.write(
                        f"This is secagg sampling {sample_num} dropping out {dropout_value*5}% with vector size {vector_dimension} try {i} \n")
                    f.close()
                    f = open("logclient.txt", "a")
                    f.write(
                        f"This is secagg sampling {sample_num} dropping out {dropout_value*5}% with vector size {vector_dimension} try {i} \n")
                    f.close()
                    # sec_agg_test.test_start_simulation(
                    #     sample_num=sample_num, share_num=sample_num, threshold=5, vector_dimension=vector_dimension, dropout_value=dropout_value, num_rounds=1)
                    light_sec_agg_test.test_start_simulation(
                         sample_num=sample_num,
                         T=int(sample_num * 0.1), U=int(sample_num * 0.7), vector_dimension=vector_dimension, dropout_value=dropout_value
                    )


'''# TESTING
vector = sec_agg_primitives.weights_zero_generate(
    [(2, 3), (2, 3)])
for i in ask_vectors_results:
    vector = sec_agg_primitives.weights_addition(vector, parameters_to_weights(
        i[1].parameters))
vector = sec_agg_primitives.weights_mod(vector, mod_range)
vector = sec_agg_primitives.weights_divide(vector, len(ask_vectors_results))
print(vector)
print(sec_agg_primitives.reverse_quantize(vector, clipping_range, target_range))
# TESTING END'''
