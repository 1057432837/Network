import numpy as np
import createDataAndPlot as cp
import copy
import math

np.seterr(all='ignore')

NETWORK_SHAPE = [2, 100, 200, 50, 2]
BATCH_SIZE = 30

LEARNING_RATE = 0.02
force_train = False
random_train = False
n_improved = 0
n_not_improved = 0


# 权重
def creat_weight(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)


# 偏置函数
def creat_biases(n_neurons):
    return np.random.randn(n_neurons)


def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 1, 1 / max_number)
    norm = array * scale_rate
    return norm


def vector_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = np.where(max_number == 0, 1, 1 / max_number)
    norm = array * scale_rate
    return norm


# 激活函数
def activation_ReLU(inputs):
    return np.maximum(0, inputs)


def activation_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)
    slided_inputs = inputs - max_values
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    norm_values = exp_values / norm_base
    return norm_values


def classify(probabilities):
    classification = np.rint(probabilities[:, 1])
    return classification


# 损失函数
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted * real_matrix, axis=1)
    return 1 - product


def loss_function(predicted, real):
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition, 1, 0)
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted * real_matrix, axis=1)
    return 1 - product


# 需求函数
def get_final_layer_preAct_demands(predicted_values, target_vector):
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector
    for i in range(len(target_vector)):
        if np.dot(target[i], predicted_values[i]) > 0.5:
            target[i] = np.array([0, 0])
        else:
            target[i] = (target[i] - 0.5) * 2
    return target


# 一层
class Layer:
    # 创建一层权重和偏置值
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = creat_weight(n_inputs, n_neurons)
        self.biases = creat_biases(n_neurons)

    # 一层的运算
    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def layer_backward(self, preWeights_values, afterWeights_demands):
        preWeight_demands = np.dot(afterWeights_demands, self.weights.T)

        condition = (preWeights_values > 0)
        value_derivatives = np.where(condition, 1, 0)
        preActs_demands = value_derivatives * preWeight_demands
        norm_preActs_demands = normalize(preActs_demands)

        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, afterWeights_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)

        return (norm_preActs_demands, norm_weight_adjust_matrix)

    def get_weight_adjust_matrix(self, preWeight_values, aftWeight_values_demands):
        plain_weight = np.full(self.weights.shape, 1)
        weights_adjust_matrix = np.full(self.weights.shape, 0.0)
        plain_weight_T = plain_weight.T
        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weight_T * preWeight_values[i, :]).T * aftWeight_values_demands[i, :]
        weights_adjust_matrix = weights_adjust_matrix / BATCH_SIZE
        return weights_adjust_matrix


# 整个神经网络
class NetWork:
    # 建立多个层
    def __init__(self, network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i + 1])
            print("\n--------------------第{}层网络构建完毕--------------------\n".format(i + 1), layer)
            self.layers.append(layer)

    # 多层运算
    def network_forward(self, inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers) - 1:
                layer_output = activation_ReLU(layer_sum)
                layer_output = normalize(layer_output)
            else:
                layer_output = activation_softmax(layer_sum)
            print("\n------------第{}层神经元经过第{}层网络到第{}层神经元------------\n".format(i + 1, i + 1, i + 2),
                  layer_output)
            outputs.append(layer_output)

        return outputs

    def network_backward(self, layer_outputs, target_vector):
        backup_network = copy.deepcopy(self)
        preAct_demands = get_final_layer_preAct_demands(layer_outputs[-1], target_vector)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - i - 1]
            if i != 0:
                layer.biases += LEARNING_RATE * np.mean(preAct_demands, axis=0)
                layer.biases = vector_normalize(layer.biases)
            outputs = layer_outputs[len(layer_outputs) - i - 2]
            results_list = layer.layer_backward(outputs, preAct_demands)
            preAct_demands = results_list[0]
            weight_adjust_matrix = results_list[1]
            layer.weights += LEARNING_RATE * weight_adjust_matrix
            layer.weights = normalize(layer.weights)
        return backup_network

    def one_batch_train(self, batch):
        global force_train, random_train, n_improved, n_not_improved
        inputs = batch[:, (0, 1)]  # 输入
        targets = copy.deepcopy(batch[:, 2]).astype(int)  # 结果
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_function(outputs[-1], targets)
        loss = loss_function(outputs[-1], targets)

        if np.mean(precise_loss) <= 0.1:
            print("No Need For Training")
        else:
            backup_network = self.network_backward(outputs, targets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], targets)
            backup_loss = loss_function(backup_outputs[-1], targets)
            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print("Improved")
                n_improved += 1
            else:
                if force_train:
                    for i in range(len(self.layers)):
                        self.layers[i].weights = backup_network.layers[i].weights.copy()
                        self.layers[i].biases = backup_network.layers[i].biases.copy()
                    print("Force Train")
                if random_train:
                    self.random_update()
                    print("Random Update")
                else:
                    print("No Improvement")
                n_not_improved += 1
        print("-------------------------------------------------")

    def train(self, n_entries):
        global force_train, random_train, n_improved, n_not_improved
        n_improved = 0
        n_not_improved = 0
        n_batches = math.ceil(n_entries / BATCH_SIZE)
        for i in range(n_batches):
            batch = cp.creat_data(BATCH_SIZE)
            self.one_batch_train(batch)
        improvement_rate = n_improved / (n_improved + n_not_improved)
        print("Improvement rate")
        print(format(improvement_rate, ".0%"))
        if improvement_rate < 0.1:
            force_train = True
        else:
            force_train = False
        if n_improved == 0:
            random_train = True
        else:
            random_train = False
        data = cp.creat_data(500)

        inputs = data[:, (0, 1)]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "After Training")

    def random_update(self):
        random_network = NetWork(NETWORK_SHAPE)
        for i in range(len(self.layers)):
            weight_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].biases
            self.layers[i].weights += weight_change
            self.layers[i].biases += biases_change

def main():
    data = cp.creat_data(500)  # 带标签的矩阵
    cp.plot_data(data, "Right Classification")
    use_this_network = 'n'
    while use_this_network != 'y' and use_this_network != 'Y':
        network = NetWork(NETWORK_SHAPE)
        inputs = data[:, (0, 1)]
        outputs = network.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "Choose Network")
        use_this_network = input("Use this network? (y/n): ")

    do_train = input("Train? (y/n)")
    while do_train == 'Y' or do_train == 'y' or do_train.isnumeric() == True:
        if do_train.isnumeric() == True:
            n_entries = int(do_train)
        else:
            n_entries = int(input("Enter the Number of Data Entries used to train\n"))
        network.train(n_entries)
        do_train = input("Train? (y/n)")

    # print("\n-------------------带正确结果的输入矩阵-------------------\n", data)  # 打印带标签的矩阵
    #
    # cp.plot_data(data, "Right Classification")

    inputs = data[:, (0, 1)]
    outputs = network.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:, 2] = classification
    cp.plot_data(data, "After Training")

    n_entries = int(input("Enter the Number of Data Entries used to train\n"))

    network.train(n_entries)

    # outputs = network.network_forward(inputs)
    #
    # classification = classify(outputs[-1])
    # print("\n-------------------------预测矩阵--------------------------\n", classification)  # 打印带标签的矩阵
    #
    # data[:, 2] = classification
    # print("\n--------------------带预测结果的输出矩阵---------------------\n", data)  # 打印带标签的矩阵
    #
    # loss = precise_loss_function(outputs[-1], targets)
    # print("\n-----------------------损失函数矩阵------------------------\n", loss)  # 打印带标签的矩阵
    #
    # demands = get_final_layer_preAct_demands(outputs[-1], targets)
    # print("\n-------------------------需求矩阵--------------------------\n", demands)
    #
    # adjust_matrix = network.layers[-1].get_weight_adjust_matrix(outputs[-2], demands)
    # print("\n-------------------------调整矩阵--------------------------\n", adjust_matrix)
    #
    # layer_backward = network.layers[-1].layer_backward(outputs[-2], demands)
    # print("\n-------------------------反向传播矩阵--------------------------\n", layer_backward)
    #
    # cp.plot_data(data, "Before training")
    #
    # backup_network = network.network_backward(outputs, targets)
    # new_outputs = backup_network.network_forward(inputs)
    # new_classification = classify(new_outputs[-1])
    # data[:, 2] = new_classification
    # cp.plot_data(data, "After training")


main()
