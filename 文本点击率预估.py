# coding:utf-8
import numpy as np 
class NeuralNetwork(): 
 # 随机初始化权重
 def __init__(self): 
      np.random.seed(1)
      self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
     # 定义激活函数：这里使用sigmoid
 def sigmoid(self, x): 
      return 1 / (1 + np.exp(-x))
 #计算Sigmoid函数的偏导数 
 def sigmoid_derivative(self, x): 
      return x * (1 - x)
 # 训练模型 
 def train(self, training_inputs, training_outputs,learn_rate, training_iterations): 
      # 迭代训练
      for iteration in range(training_iterations):
           #前向计算
           output = self.think(training_inputs)
           # 计算误差
           error = training_outputs - output
           # 反向传播-BP-微调权重
           adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
           self.synaptic_weights += learn_rate*adjustments
 def think(self, inputs): 
      # 输入通过网络得到输出
      # 转化为浮点型数据类型
      inputs = inputs.astype(float)
      output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
      return output
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("随机初始化的权重矩阵W")
    print(neural_network.synaptic_weights)



train_data=[[0,0,1], [1,1,1], [1,0,1], [0,1,1]]
training_inputs = np.array(train_data)
 # 模拟训练数据Y
training_outputs = np.array([[0,1,1,0]]).T
 # 定义模型的参数：
# 参数学习率
learn_rate=0.1
# 模型迭代的次数
epoch=150000
neural_network.train(training_inputs, training_outputs, learn_rate, epoch)
print("迭代计算之后权重矩阵W: ")
print(neural_network.synaptic_weights)
# 模拟需要预测的数据X
pre_data=[0,0,1]
 # 使用训练的模型预测该微博被点击的概率
print("该微博被点击的概率：")
print(neural_network.think(np.array(pre_data)))
