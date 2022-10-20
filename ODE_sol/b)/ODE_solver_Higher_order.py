#ODE solver
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
import math
from matplotlib import pyplot as plt
import numpy as np
class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        #super viene de "clase superior", o "clase padre". Se tiene que comportar
        #como la función __init__ de la función padre.
        super().__init__(**kwargs)
        #Nuestra clase es una variable que va a ir midiendo nuestra función de costo (definida por nosotros.)
        self.loss_trackers = keras.metrics.Mean(name='loss')
    
    #Siempre se tiene que hacer para que cuando entrene e imprima el valor del a función de costo, sepa qué variable
    #imprimir. Esta siempre tiene que llamarse metrics.
    @property
    def metrics(self):
        return [self.loss_trackers]
    
    #La función train_step ya viene definida en el modelo Sequential. Al definirla nuevamente y cambiarla, 
    #el modelo Sequential va a cambiar esta variable y se reemplazará por lo que escribamos abajo. 
    def train_step(self, data):
        #El batch_size puede simplemente no definirse. La bondad de haberlo hecho es que desde aquí podemos cambiar
        #la resolución con que queremos aproximar la solución a la EDO. 
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval=-5, maxval=5)
        x_o = tf.zeros((batch_size, 1))

#Acá creo que debo hacer el tf.GradientTape(persistent=True), para calcular derivadas de orden más alto. 
        with tf.GradientTape() as tape:
            '''Compute the loss value'''
            with tf.GradientTape() as tape3:
                tape3.watch(x)

                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(x)
                    tape2.watch(x_o)
                    y_pred = self(x, training=True)
                    y_o = self(x_o, training=True)
                dy = tape2.gradient(y_pred, x)
                #print(dy)  #Este me da un tensor
                dy_o = tape2.gradient(y_o, x_o)
                #print(dy_o)  #Este me daba un None por no poner el watch.
            
                
                

            #Para cuando la red necesita una segunda derivada
            dyy = tape3.gradient(dy, x)  #Este también me da none
            print('Aquí imprimí algo')
            print(dyy)
            #Aquí usé la otra condición inicial
            ic2 = dy_o + 0.5
            
           
            #Esta es la ecuación que estamos resolviendo
            #Queremos que la variable eq llegue a cero, pues eso es lo que dice la EDO.
            eq = dyy + y_pred 
            #Esta es la condición inicial. También queremos que ic sea cero.
            ic = y_o - 1.0
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic) + keras.losses.mean_squared_error(0., ic2)
        
        '''Apply grads'''
        grads = tape.gradient(loss, self.trainable_variables)
        #self.trainable_variables es la variable que queremos actualizar.
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        '''Update metrics'''
        self.loss_trackers.update_state(loss)
        '''Return a dict mapping metric names to current value'''
        return {"loss": self.loss_trackers.result()}

model = ODEsolver()

#¿¿Por qué no tiene segunda entrada la input_shape??
model.add(Dense(10, activation = 'tanh', input_shape=(1,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
#Necesitamos la activación lineal porque la tanh no devuelve valores de R, devuelve valores de un conjunto
#más restringido. Con la activación lineal, ampliamos los valores que puede devolver la red para que satisfaga
#nuestras necesidades. 
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer=RMSprop(), metrics=['loss'])

x=tf.linspace(-5, 5, 100)
history = model.fit(x, epochs=1500, verbose=1)

x_testv = tf.linspace(-5, 5, 100)
a = model.predict(x_testv)
plt.plot(x_testv, a, label = 'Función numérica')

#Expresión analítica para la solución
plt.plot(x_testv, -0.5*np.sin(x)+np.cos(x), label = 'Función analítica -0.5sin(x)+cos(x)')
plt.legend(loc = 'upper left')
plt.title('Solución a la EDO d^2y/dx^2=-y, con condiciones iniciales y(0)=1, dy/dx(0)=-0.5')
plt.show()


'''Para guardar el modelo en disco'''
model.save("red.h5")
exit()
'''Para cargar la red'''
modelo_cargado = tf.keras.models.load_model('red.h5')