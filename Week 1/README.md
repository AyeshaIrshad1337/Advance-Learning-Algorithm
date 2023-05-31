# Neural Network Intuition
<details>
  <summary>Neurons and the brain</summary>
  
  When neural networks were first invented many decades ago, the original motivation was to write software that could mimic how the human brain or how the biological brain learns and thinks. Even though today, neural networks, sometimes also called artificial neural networks, have become very different than how any of us might think about how the brain actually works and learns. Some of the biological motivations still remain in the way we think about artificial neural networks or computer neural networks today. Let's start by taking a look at how the brain works and how that relates to neural networks.   
    
Even though today's neural networks have almost nothing to do with how the brain learns, there was the early motivation of trying to build software to mimic the brain. So how does the brain work? Here's a diagram illustrating what neurons in a brain look like. 
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/f51a09f7-a3b3-4cc4-9a82-f8d0cab2956d)

    
  All of human thought is from neurons like this in your brain and mine, sending electrical impulses and sometimes forming new connections of other neurons. Given a neuron like this one, it has a number of inputs where it receives electrical impulses from other neurons, and then this neuron that I've circled carries out some computations and will then send this outputs to other neurons by this electrical impulses, and this upper neuron's output in turn becomes the input to this neuron down below, which again aggregates inputs from multiple other neurons to then maybe send its own output, to yet other neurons, and this is the stuff of which human thought is made. 
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/dd72218b-9b72-4339-ab16-b550bbed6590)
  
  
  Here's a simplified diagram of a biological neuron. A neuron comprises a cell body shown here on the left, and if you have taken a class in biology, you may recognize this to be the nucleus of the neuron. As we saw on the previous slide, the neuron has different inputs. In a biological neuron, the input wires are called the dendrites, and it then occasionally sends electrical impulses to other neurons via the output wire, which is called the axon. Don't worry about these biological terms. If you saw them in a biology class, you may remember them, but you don't really need to memorize any of these terms for the purpose of building artificial neural networks. But this biological neuron may then send electrical impulses that become the input to another neuron. So the artificial neural network uses a very simplified Mathematical model of what a biological neuron does. I'm going to draw a little circle here to denote a single neuron. What a neuron does is it takes some inputs, one or more inputs, which are just numbers. It does some computation and it outputs some other number, which then could be an input to a second neuron, shown here on the right. When you're building an artificial neural network or deep learning algorithm, rather than building one neuron at a time, you often want to simulate many such neurons at the same time. In this diagram, I'm drawing three neurons. What these neurons do collectively is input a few numbers, carry out some computation, and output some other numbers.  
  ### Why is it that only in the last handful of years that neural networks have really taken off?
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/fe6de8e1-bd09-4613-83c5-42169d6dbb69)

  Let see the plot on the horizontal axis the amount of data you have for a problem, and on the vertical axis, the performance or the accuracy of a learning algorithm applied to that problem.  
  Due the rise of data thanks to internet and rise of processors i.e Faster computer process GPU. Now we can use large neural network on large data which will give us a high performance.

</details>

# TensorFlow Implementation 
<details>
  <summary>Inference in Code</summary>
  
 + TensorFlow is one of the leading frameworks to implementing deep learning algorithms. 
  + The other popular tool is PyTorch. But we're going to focus in this specialization on TensorFlow. 
  ## how you can implement inferencing code using TensorFlow. Let's dive in. 
  One of the remarkable things about neural networks is the same algorithm can be applied to so many different applications. 
    
  Lets use an example to illustrate inference. Sometimes I do like to roast coffee beans myself at home. My favorite is actually Colombian coffee beans. Can the learning algorithm help optimize the quality of the beans you get from a roasting process like this? When you're roasting coffee, two parameters you get to control are the temperature at which you're heating up the raw coffee beans to turn them into nicely roasted coffee beans, as well as the duration or how long are you going to roast the beans.  
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/6762752f-01b1-47b9-b67a-1d599a5f1487)
  
  In this slightly simplified example, we've created the datasets of different temperatures and different durations, as well as labels showing whether the coffee you roasted is good-tasting coffee. Where cross here, the positive cross y equals 1 corresponds to good coffee, and all the negative cross corresponds to bad coffee. It looks like a reasonable way to think of this dataset is if you cook it at too lower temperature, it doesn't get roasted and it ends up undercooked. If you cook it, not for long enough, the duration is too short, it's also not a nicely roasted set of beans. Finally, if you were to cook it either for too long or for too higher temperature, then you end up with overcooked beans. They're a little bit burnt beans. There's not good coffee either. It's only points within this little triangle here that corresponds to good coffee. 
  
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/37c0dfff-34d3-41ab-90c2-81aaa115392e)
  
   This example is simplified a bit from actual coffee roasting. Even though this example is a simplified one for the purpose of illustration, there have actually been serious projects using machine learning to optimize coffee roasting as well. The task is given a feature vector x with both temperature and duration, say 200 degrees Celsius for 17 minutes, how can we do inference in a neural network to get it to tell us whether or not this temperature and duration setting will result in good coffee or not?   
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/d88f2418-670c-4302-84ca-9ddc85c383ec)
  
  We're going to set x to be an array of two numbers. The input features 200 degrees celsius and 17 minutes. Then you create Layer 1 as this first hidden layer, the neural network, as dense open parenthesis units 3, that means three units or three hidden units in this layer using as the activation function, the sigmoid function. Dense is another name for the layers of a neural network that we've learned about so far. As you learn more about neural networks, you learn about other types of layers as well. But for now, we'll just use the dense layer, which is the layer type you've learned about in the last few lectures for all of our examples. Next, you compute a1 by taking Layer 1, which is actually a function, and applying this function Layer 1 to the values of x. That's how you get a1, which is going to be a list of three numbers because Layer 1 had three units. So a1 here may, just for the sake of illustration, be 0.2, 0.7, 0.3. 
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/3e149c68-0055-4cd8-bc1d-d4e543dc8249)
  
  
  Next, for the second hidden layer, Layer 2, would be dense. Now this time it has one unit and again to sigmoid activation function, and you can then compute a2 by applying this Layer 2 function to the activation values from Layer 1 to a1. That will give you the value of a2, which for the sake of illustration is maybe 0.8. Finally, if you wish to threshold it at 0.5, then you can just test if a2 is greater and equal to 0.5 and set y-hat equals to one or zero positive or negative cross accordingly. That's how you do inference in the neural network using TensorFlow. 
  
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/e0d0c6da-2c55-4c5b-821a-daf4d5ef0d09)
  
  There are some additional details that I didn't go over here, such as how to load the TensorFlow library and how to also load the parameters w and b of the neural network. But we'll go over that in the lab. Please be sure to take a look at the lab. But these are the key steps for forward propagation in how you compute a1 and a2 and optionally threshold a2. 
    
####  Let's look at one more example 
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/665c1ac3-dd5b-4a50-b01a-0a5e2a7ce015)

 In this example, x is a list of the pixel intensity values. So x is equal to a numpy array of this list of pixel intensity values. Then to initialize and carry out one step of forward propagation, Layer 1 is a dense layer with 25 units and the sigmoid activation function. You then compute a1 equals the Layer 1 function applied to x. To build and carry out inference through the second layer, similarly, you set up Layer 2 as follows, and then computes a2 as Layer 2 applied to a1. Then finally, Layer 3 is the third and final dense layer. Then finally, you can optionally threshold a3 to come up with a binary prediction for y-hat. 
    
  ![image](https://github.com/AyeshaIrshad1337/Advance-Learning-Algorithm/assets/104616632/3bfab1b2-4532-4be6-9039-d1587079a7ce)

    
  That's the syntax for carrying out inference in TensorFlow. One thing I briefly alluded to is the structure of the numpy arrays. TensorFlow treats data in a certain way that is important to get right.
</details>
