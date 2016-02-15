#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # All parameters are list of lists in default, have to be converted to lists
    errors = predictions - net_worths
    errors = [val for sublist in errors for val in sublist]
    ages = [val for sublist in ages for val in sublist]
    net_worths = [val for sublist in net_worths for val in sublist]
    predictions = [val for sublist in predictions for val in sublist]
    
    
    cleaned_data = []
    
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({'age':ages,
                       'prediction':predictions,
                       'net_worth':net_worths,
                       'error':errors})
                    
    p = np.percentile(np.absolute(df['error']), 90)
    df = df[np.absolute(df['error'])<=p]
    
    cleaned_data.append(zip(df['age'], df['net_worth'], df['error']))
    cleaned_data = [val for sublist in cleaned_data for val in sublist]
   
    return cleaned_data

