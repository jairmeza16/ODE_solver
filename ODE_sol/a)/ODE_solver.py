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

#Acá creo que debo hacer el tf.GradientTape(persistent=True), para calcular derivadas de orden más alto. 
        with tf.GradientTape() as tape:
            '''Compute the loss value'''
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)

            x_o = tf.zeros((batch_size, 1))
            y_o = self(x_o, training=True)
            #Esta es la ecuación que estamos resolviendo
            #Queremos que la variable eq llegue a cero, pues eso es lo que dice la EDO.
            eq = x*dy + y_pred - x**2*tf.math.cos(x) 
            #Esta es la condición inicial. También queremos que ic sea cero.
            ic = y_o - 0.
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)
        
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
#Aumenté el número de neuronas de 1 a 10. 
#Añadí otras capas de neuronas...
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
#Necesitamos la activación lineal porque la tanh no devuelve valores de R, devuelve valores de un conjunto
#más restringido. Con la activación lineal, ampliamos los valores que puede devolver la red para que satisfaga
#nuestras necesidades. 
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer=RMSprop(), metrics=['loss'])

x=tf.linspace(-5, 5, 200)
history = model.fit(x, epochs=1500, verbose=1)

x_testv = tf.linspace(-5, 5, 200)
a = model.predict(x_testv)
plt.plot(x_testv, a, label = 'Función numérica')

#Expresión analítica para la solución
plt.plot(x_testv, x*np.sin(x)-(2/x)*(np.sin(x)-x*np.cos(x)), label = 'Función analítica')
plt.legend(loc = 'upper left')
plt.plot(x_testv, 0*x, label = 'y=0')
plt.title('Solución a la ecuación diferencial xy+y=x^2cos(x) para la condición inicial y(0)=0')
plt.show()


'''Para guardar el modelo en disco'''
model.save("red.h5")
exit()

'''Para cargar la red'''
modelo_cargado = tf.keras.models.load_model('red.h5')