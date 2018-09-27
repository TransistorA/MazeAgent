# Some potentially useful modules
# Whether or not you use these (or others) depends on your implementation!
import random
import numpy
import math
import matplotlib.pyplot as plt


class NeuralMMAgent(object):
    """
    Class to for Neural Net Agents
    """

    def __init__(self, num_in_nodes, num_hid_nodes, num_hid_layers, num_out_nodes,
                 learning_rate=0.2, max_epoch=10000, max_sse=.1, momentum=0.7, random_seed=1):
        """
        Arguments:
            num_in_nodes -- total # of input layers for Neural Net
            num_hid_nodes -- total # of hidden nodes for each hidden layer
                in the Neural Net
            num_hid_layers -- total # of hidden layers for Neural Net
            num_out_nodes -- total # of output layers for Neural Net
            learning_rate -- learning rate to be used when propagating error
            creation_function -- function that will be used to create the
                neural network given the input
            activation_function -- list of two functions:
                1st function will be used by network to determine activation given a weighted summed input
                2nd function will be the derivative of the 1st function
            random_seed -- used to seed object random attribute.
                This ensures that we can reproduce results if wanted
        """
        assert num_in_nodes > 0 and num_hid_layers > 0 and num_hid_nodes and\
            num_out_nodes > 0, "Illegal number of input, hidden, or output layers!"

        self.num_in_nodes = num_in_nodes
        self.num_hid_nodes = num_hid_nodes
        self.num_hid_layers = num_hid_layers
        self.num_out_nodes = num_out_nodes
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.max_sse = max_sse
        self.momentum = momentum
        self.random_seed = random_seed
        self.activation_function = self.sigmoid_af

        # values to be set
        self.input_list = []
        self.output_list = []

        # Generate random weight list
        weight_list = []
        input_weight = []
        for i in range(self.num_in_nodes * self.num_hid_nodes):
            input_weight.append(random.uniform(-.5, .5))
        weight_list.append(input_weight)
        for i in range(self.num_hid_layers - 1):
            hidden_weight = []
            for j in range(self.num_hid_nodes * self.num_hid_nodes):
                hidden_weight.append(random.uniform(-.5, .5))
            weight_list.append(hidden_weight)
        output_weight = []
        for i in range(self.num_hid_nodes * self.num_out_nodes):
            output_weight.append(random.uniform(-.5, .5))
        weight_list.append(output_weight)
        self.weight_list = weight_list

        # Generate initial thetas of all zeros
        theta_list = [[0] * self.num_in_nodes]
        for i in range(self.num_hid_layers):
            hidden_theta = [0] * self.num_hid_nodes
            theta_list.append(hidden_theta)
        theta_list.append([0] * self.num_out_nodes)
        self.theta_list = theta_list

    def train_net(self, input_list, output_list, max_num_epoch=100000, max_sse=0.1):
        """ Trains neural net using incremental learning
            (update once per input-output pair)
            Arguments:
                input_list -- 2D list of inputs
                output_list -- 2D list of outputs matching inputs
        """
        assert len(input_list[0]) == self.num_in_nodes and len(output_list[0]) == self.num_out_nodes and \
               len(input_list) == len(output_list), "Illegal input or output number"

        self.input_list = input_list
        self.output_list = output_list

        all_err = []

        previous_weight_deltas = []  # Initial set of the previous weight deltas
        if self.num_hid_layers >= 1:
            previous_weight_deltas.append([0] * self.num_in_nodes * self.num_hid_nodes)
            for i in range(self.num_hid_layers - 1):
                previous_weight_deltas.append([0] * self.num_hid_nodes * self.num_hid_nodes)
            previous_weight_deltas.append([0] * self.num_hid_nodes * self.num_out_nodes)
        else:
            previous_weight_deltas.append([0] * self.num_in_nodes * self.num_out_nodes)

        num_epoch = 0
        while True:
            total_err = 0
            for i in range(len(input_list)):
                neuron_value_list = self.calculate_output(input_list[i])
                errors = self.calculate_errors(neuron_value_list, output_list[i])
                self.update_weights(neuron_value_list, errors, previous_weight_deltas)
                previous_weight_deltas = self.calculate_weight_delta(neuron_value_list, errors)  # The calculation has
                # nothing to do with weights, so we can put it after the weight update
                total_err += self.compute_output_error(output_list[i], neuron_value_list[-1])

                if i == 0 and num_epoch == 0:
                    print("The weights after 1st step for input [1,0] are ", self.weight_list)
                    print("The weight deltas are ", previous_weight_deltas)
                    print("And the outputs are ", neuron_value_list)

            total_err = total_err / len(input_list)
            all_err.append(total_err)
            num_epoch += 1

            # print(num_epoch, "  ", total_err)

            if num_epoch >= max_num_epoch:
                break

            if total_err < max_sse:
                break

        print("\nThe weights after ", num_epoch, " epochs are ", self.weight_list)
        print("The outputs for the input values are:")
        for i in range(len(input_list)):
            print(input_list[i], "  ", self.calculate_output(input_list[i])[-1])

        # Show how the error has changes
        plt.plot(all_err)
        plt.show()

    def set_weights(self, weight_list):
        # exit when number of weights is illegal
        assert len(weight_list) == self.num_hid_layers + 1 and len(weight_list[0]) == self.num_hid_nodes * \
               self.num_in_nodes and len(weight_list[-1]) == self.num_hid_nodes * \
               self.num_out_nodes, "Illegal number of weights for input or output layer"
        for i in range(1, len(weight_list)-1):  # the number of weights between hidden layers
            if len(weight_list[i]) != self.num_hid_nodes * self.num_hid_nodes:
                print("Illegal number of weights between hidden layers")
                exit(0)
        self.weight_list = weight_list

    def set_thetas(self, theta_list):
        # exit when number of thetas is illegal
        assert len(theta_list) == self.num_hid_layers + 2 and len(theta_list[0]) == self.num_in_nodes and \
               len(theta_list[-1]) == self.num_out_nodes, "Illegal number of thetas for input or output layer"
        for i in range(1, len(theta_list)-1):  # the number of thetas for each hidden layer
            if len(theta_list[i]) != self.num_hid_nodes:
                print("Illegal number of thetas for hidden layers")
                exit(0)
        self.theta_list = theta_list

    def calculate_output(self, input_values):  # output the values of all neurons according to the input and weights
        assert len(input_values) == self.num_in_nodes, "Illegal number of input or output"
        assert len(self.theta_list) == self.num_hid_layers + 2 and len(self.theta_list[0]) == self.num_in_nodes and \
               len(self.theta_list[-1]) == self.num_out_nodes, "Illegal number of thetas for input or output layer"
        neuron_value_list = [input_values]
        for i in range(len(self.weight_list)-1):  # number of hidden layers
            hidden_layer = []
            for j in range(self.num_hid_nodes):  # number of hidden nodes
                neuron_value = 0  # calculate value of each hidden node
                for k in range(len(neuron_value_list[i])): # number of nodes in previous layer
                    neuron_value += neuron_value_list[i][k] * self.weight_list[i][self.num_hid_nodes * k + j]
                hidden_layer.append(self.activation_function(neuron_value - self.theta_list[i + 1][j]))  # g(sum(xw) - theta)
            neuron_value_list.append(hidden_layer)

        output_layer = []  # calculate the output layer
        for i in range(self.num_out_nodes):
            neuron_value = 0
            for j in range(len(neuron_value_list[-1])):
                neuron_value += neuron_value_list[-1][j] * self.weight_list[-1][self.num_out_nodes * j + i]
            output_layer.append(self.activation_function(neuron_value))
        neuron_value_list.append(output_layer)
        return neuron_value_list

    def calculate_errors(self, neuron_value_list, expected_output):  # Calculate errors by back propagation
        assert len(expected_output) == len(neuron_value_list[-1]) and len(expected_output) == self.num_out_nodes and \
               len(neuron_value_list[0]) == self.num_in_nodes, "Illegal number of inputs or outputs"

        output_errors = []  # Calculate the errors for the output layer
        for i in range(self.num_out_nodes):
            output = neuron_value_list[-1][i]
            output_error = output * (1 - output) * (expected_output[i] - output)
            output_errors.append(output_error)
        errors = [output_errors]

        # Calculate the errors for hidden layer
        for i in range(1, self.num_hid_layers + 1):  # the last i hidden layer
            hidden_errors = []
            for j in range(self.num_hid_nodes):  # the number of nodes in current hidden layer
                hidden_error_sum = 0
                for k in range(int(len(self.weight_list[-i])/self.num_hid_nodes)):  # the number of nodes of next layer
                    hidden_error_sum += errors[0][k] * self.weight_list[-i][int(j * len(self.weight_list[-i])
                                                                                / self.num_hid_nodes + k)]
                neuron_value = neuron_value_list[self.num_hid_layers-i+1][j]
                neuron_error = neuron_value * (1 - neuron_value) * hidden_error_sum
                hidden_errors.append(neuron_error)
            errors = [hidden_errors] + errors
        return errors

    def calculate_weight_delta(self, neuron_value_list, errors):
        # Calculate the weight delta according to the neuron values and the error list. It doesn't change weight list.
        weight_deltas = []  # Initial set of the weight deltas. Set all deltas to 0.
        if self.num_hid_layers >= 1:
            weight_deltas.append([0] * self.num_in_nodes * self.num_hid_nodes)
            for i in range(self.num_hid_layers - 1):
                weight_deltas.append([0] * self.num_hid_nodes * self.num_hid_nodes)
            weight_deltas.append([0] * self.num_hid_nodes * self.num_out_nodes)
        else:
            weight_deltas.append([0] * self.num_in_nodes * self.num_out_nodes)

        if self.num_hid_layers >= 1:
            # update the weights between input and 1st hidden layer
            for i in range(self.num_in_nodes):
                for j in range(self.num_hid_nodes):
                    weight_change = self.learning_rate * neuron_value_list[0][i] * errors[0][j]
                    weight_deltas[0][i * self.num_hid_nodes + j] = weight_change

            # update the weights between last hidden and output layer
            for i in range(self.num_hid_nodes):
                for j in range(self.num_out_nodes):
                    weight_change = self.learning_rate * neuron_value_list[-2][i] * errors[-1][j]
                    weight_deltas[-1][i * self.num_out_nodes + j] = weight_change

            # update the weights between hidden layers
            for k in range(self.num_hid_layers - 1):
                for i in range(self.num_hid_nodes):
                    for j in range(self.num_hid_nodes):
                        weight_change = self.learning_rate * neuron_value_list[k+1][i] * errors[k+1][j]
                        weight_deltas[k+1][i * self.num_hid_nodes + j] = weight_change
        else:  # there is no hidden layer
            for i in range(self.num_in_nodes):
                for j in range(self.num_out_nodes):
                    weight_change = self.learning_rate * neuron_value_list[1][i] * errors[0][j]
                    weight_deltas[0][i * self.num_out_nodes + j] = weight_change
        return weight_deltas

    def update_weights(self, neuron_value_list, errors, previous_weight_deltas):  # Update weights and thetas
        # Update the thetas
        for i in range(self.num_hid_layers + 1):  # the number of layers expect the input layer
            for j in range(len(self.theta_list[i + 1])):  # the number of nodes in each layer
                self.theta_list[i + 1][j] -= self.learning_rate * errors[i][j]  # delta theta(j) = -alpha * error(j)

        if self.num_hid_layers >= 1:
            # update the weights between input and 1st hidden layer
            for i in range(self.num_in_nodes):
                for j in range(self.num_hid_nodes):
                    weight_change = self.learning_rate * neuron_value_list[0][i] * errors[0][j]
                    self.weight_list[0][i * self.num_hid_nodes + j] += weight_change + self.momentum * previous_weight_deltas[0][i * self.num_hid_nodes + j]

            # update the weights between last hidden and output layer
            for i in range(self.num_hid_nodes):
                for j in range(self.num_out_nodes):
                    weight_change = self.learning_rate * neuron_value_list[-2][i] * errors[-1][j]
                    self.weight_list[-1][i * self.num_out_nodes + j] += weight_change + self.momentum * previous_weight_deltas[-1][i * self.num_out_nodes + j]

            # update the weights between hidden layers
            for k in range(self.num_hid_layers - 1):
                for i in range(self.num_hid_nodes):
                    for j in range(self.num_hid_nodes):
                        weight_change = self.learning_rate * neuron_value_list[k+1][i] * errors[k+1][j]
                        self.weight_list[k+1][i * self.num_hid_nodes + j] += weight_change + self.momentum * previous_weight_deltas[k+1][i * self.num_hid_nodes + j]

        else:  # there is no hidden layer
            for i in range(self.num_in_nodes):
                for j in range(self.num_out_nodes):
                    weight_change = self.learning_rate * neuron_value_list[1][i] * errors[0][j]
                    self.weight_list[0][i * self.num_out_nodes + j] += weight_change + self.momentum * previous_weight_deltas[0][i * self.num_out_nodes + j]

    @staticmethod
    def compute_output_error(expected_list, actual_output_list):
        assert len(actual_output_list) == len(expected_list), "Illegal length of training list"
        error = 0;
        for i in range(len(actual_output_list)):
            error += abs(expected_list[i]-actual_output_list[i])
        return error / len(actual_output_list)

    @staticmethod
    def sigmoid_af(summed_input):
        return 1 / (1 + math.e**(-summed_input))

    @staticmethod
    def sigmoid_af_deriv(sig_output):
        return 1 - sig_output


test_agent = NeuralMMAgent(2, 2, 2, 1, random_seed=5, max_epoch=100000, learning_rate=0.2, momentum=0.7)  # Network with 2 hidden layers

# We can comment the previous line and uncomment the next 3 lines to test the in class set of weights
#test_agent = NeuralMMAgent(2, 2, 1, 1, random_seed=5, max_epoch=100000, learning_rate=0.2, momentum=0.7)  # Network with 1 hidden layer
#test_agent.set_weights([[-.37, .26, .1, -.24], [-.01, -.05]])  # We can comment out this line to generate random weights
#test_agent.set_thetas([[0, 0], [0, 0], [0]])

test_in = [[1, 0], [0, 0], [1, 1], [0, 1]]
test_out = [[1], [0], [0], [1]]

test_agent.train_net(test_in, test_out, max_sse=test_agent.max_sse, max_num_epoch=test_agent.max_epoch)
