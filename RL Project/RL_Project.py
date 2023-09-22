# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import math 
import time
from tqdm import tqdm
  
# Define Action class
class Actions:
  def __init__(self, actionNumber):
    self.actionNumber = actionNumber
    self.quality =np.random.uniform(0.0,1.0)   
    self.softmaxDis=0.0
    self.N = 0
  
  
  def giveReward(self): 
    return np.random.uniform(self.quality,1.0)
  
 
  def update(self, reward):
    self.N += 1
    self.quality = (1 - 1.0 / self.N)*self.quality + 1.0 / self.N * reward    
    

class RL:
  def __init__(self,actionsCount,steps,iterationsCount):
      self.actionsCount=actionsCount
      self.steps=steps
      self.iterationsCount=iterationsCount
      self.actions=[]
      for index in range(1,actionsCount+1):          
          self.actions.append(Actions(index))

  def greedy(self): 
       data = np.empty(self.steps)
       selection = np.empty(self.steps)
       initilaActionsValue=self.actions
       optimalActionIndex=np.argmax([a.quality for a in self.actions])
       for step in range(self.steps):
           data[step]=0
           selection[step]=0
       print("Please Wait,it maybe takes a few minutes. start Greedy approach \n")
       for iteration in tqdm(range(self.iterationsCount)):
           #self.showProgress(iteration)
           self.actions=initilaActionsValue
           for step in range(self.steps):
               j = np.argmax([a.quality for a in self.actions])
               reward = self.actions[j].giveReward()
               self.actions[j].update(reward)
               data[step] += reward
               if(j == optimalActionIndex):
                   selection[step]+=1
       for step in range(self.steps):
           data[step]/=self.iterationsCount
           selection[step]=selection[step]*100/self.iterationsCount             
       cumulative_average = np.cumsum(data) / (np.arange(self.steps) + 1)
       cumulative_selection = np.cumsum(selection) / (np.arange(self.steps) + 1)
       plt.plot(cumulative_average)
       plt.title("Average Of Rewards (Greedy)")
       plt.xscale('log')
       plt.show() 

       plt.plot(selection)
       plt.title("Optimal Action% (Greedy)")
       plt.xscale('log')
       plt.show()        

       return cumulative_average   

  def epsilonGreedy(self): 
       eps=input("Enter Epsilon :")
       eps=float(eps)
       data = np.empty(self.steps)
       selection = np.empty(self.steps)
       initilaActionsValue=self.actions
       optimalActionIndex=np.argmax([a.quality for a in self.actions])
       for step in range(self.steps):
           data[step]=0
           selection[step]=0
       print("Please Wait,it maybe takes a few minutes.Start Epsilon Greedy approach \n")
       for iteration in tqdm(range(self.iterationsCount)):
           #self.showProgress(iteration)
           self.actions=initilaActionsValue
           for step in range(self.steps):
               p = np.random.random()
               if p < eps:
                   j = np.random.choice(self.actionsCount)
               else:
                   j = np.argmax([a.quality for a in self.actions])
               reward = self.actions[j].giveReward()
               self.actions[j].update(reward)
               data[step] += reward
               if(j == optimalActionIndex):
                selection[step]+=1
       for step in range(self.steps):
           data[step]/=self.iterationsCount
           selection[step]=selection[step]*100/self.iterationsCount
       cumulative_average = np.cumsum(data) / (np.arange(self.steps) + 1)
       cumulative_selection = np.cumsum(selection) / (np.arange(self.steps) + 1)
       plt.plot(cumulative_average)
       plt.title("Average Of Rewards (Epsilon Greedy) ε="+str(eps))
       plt.xscale('log')
       plt.show() 

       plt.plot(selection)
       plt.title("Optimal Action% (Epsilon Greedy) ε="+str(eps))
       plt.xscale('log')
       plt.show()        

       return cumulative_average  

  def softmax(self): 
       data = np.empty(self.steps)
       selection = np.empty(self.steps)
       initilaActionsValue=self.actions
       optimalActionIndex=np.argmax([a.quality for a in self.actions])
       for step in range(self.steps):
           data[step]=0
           selection[step]=0
       print("Please Wait,it maybe takes a few minutes.Start Softmax approach \n")
       for iteration in tqdm(range(self.iterationsCount)):
           #self.showProgress(iteration)
           self.actions=initilaActionsValue
           for step in range(self.steps):
               self.calculateSoftmaxDis(step,self.iterationsCount)
               j = np.argmax([a.softmaxDis for a in self.actions])
               reward = self.actions[j].giveReward()
               self.actions[j].update(reward)
               data[step] += reward
               if(j == optimalActionIndex):
                   selection[step]+=1
       for step in range(self.steps):
           data[step]/=self.iterationsCount
           selection[step]=selection[step]*100/self.iterationsCount
       cumulative_average = np.cumsum(data) / (np.arange(self.steps) + 1)
       cumulative_selection = np.cumsum(selection) / (np.arange(self.steps) + 1)
       plt.plot(cumulative_average)
       plt.title("Average Of Rewards (Softmax)")
       plt.xscale('log')
       plt.show() 

       plt.plot(selection)
       plt.title("Optimal Action% (Softmax)")
       plt.xscale('log')
       plt.show()        

       return cumulative_average

  def calculateSoftmaxDis(self,step,iterationsCount):     
      computationalTemp=0.1 #iterationsCount+1-step
      tempTotal=0.0
      for action in self.actions:
        tempValue=action.quality/computationalTemp
        tempValue=float(str("{:.2f}".format(tempValue)))
        tempTotal=tempTotal+math.exp(tempValue)     
     
      for action in self.actions:  
        tempValue=action.quality/computationalTemp
        tempValue=float(str("{:.2f}".format(tempValue)))
        action.softmaxDis=math.exp(tempValue)/tempTotal
      


        

if __name__ == '__main__':
  actions_Count=input("Enter Number Of Actions: ")
  actions_Count=int(actions_Count)
  stepsCount=input("Steps Count:")
  stepsCount=int(stepsCount)
  iterationsCount=input("Iterations Count:")
  iterationsCount=int(iterationsCount)
  rl=RL(actions_Count,stepsCount,iterationsCount) 
  rl.greedy()
  rl.epsilonGreedy()
  rl.softmax()

