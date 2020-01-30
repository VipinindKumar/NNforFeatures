def nn(act, plot_history=False):
    
    # define input for the model
    input = Input(shape=(X_scl.shape[1],))
    
    # hidden layers of the model
    h1 = Dense(32, activation=act, kernel_regularizer=regularizers.l2(0.03))
    a1 = h1(input)

    h2 = Dense(32, activation=act, kernel_regularizer=regularizers.l2(0.03))
    a2 = h2(a1)

    h3 = Dense(4, activation=act)
    a3 = h3(a2)
    
    # define output of the model 
    output = Dense(1, activation='sigmoid')(a3)
    
    # sets the input and output of the model
    model = Model(inputs=input, outputs=output)
    
    # define Adam optimizer
    adam_opt = Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # compile the model and define loss, optimizer and metric
    model.compile(loss='mean_squared_error', optimizer=adam_opt, metrics=['accuracy'])
    
    # fitting the model
    fit = model.fit(X_scl, Y, epochs=500, validation_split=0.3, verbose=0)
    
    # print finale values for loss and accuracy
    print('NN train loss:', fit.history['loss'][-1])
    print('NN val loss:', fit.history['val_loss'][-1])
    
    print('NN train accuracy:', fit.history['accuracy'][-1])
    print('NN val accuracy:', fit.history['val_accuracy'][-1])

    if plot_history:
        # create 2 vertical subplots
        fig, ax = plt.subplots(2, figsize=(8,12))

        # plot for loss
        ax[0].plot(fit.history['val_loss'])
        ax[0].plot(fit.history['loss'])
        ax[0].legend(['val', 'train'])
        ax[0].set(xlabel='epochs', ylabel='MSE loss')

        # plot for accuracy
        ax[1].plot(fit.history['val_accuracy'])
        ax[1].plot(fit.history['accuracy'])
        ax[1].legend(['val', 'train'])
        ax[1].set(xlabel='epochs', ylabel='Accuracy')
    
    return fit, model
