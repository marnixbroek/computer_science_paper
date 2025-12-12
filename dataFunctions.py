import json
import re

def loadData(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    flattenedData = flattenData(data)
    cleanedData = cleanData(flattenedData)
    return cleanedData

def saveDataInVariables(data):
    """Save data into variables for easier access."""
    variables = {}
    for key, value in data.items():
        variables[key] = value
    return variables

def flattenData(data):
    # Initialize the new flattened dictionary
    flattened_data = {}

    # Process the data
    for original_key, product_list in data.items():
        # Counter for multiple entries under the same key
        counter = 1
        
        # Iterate through the list of product dictionaries
        for product in product_list:
            # Create the new unique key: {original_key}_{counter}
            new_key = f"{original_key}_{counter}"
            
            # Map the new key to the individual product dictionary
            flattened_data[new_key] = product
            
            # Increment the counter
            counter += 1

    return flattened_data

def cleanData(data):
    """Perform basic cleaning on the data.
    We standardize the most frequently occurring units:
    inch and Hertz."""
    cleanedData = data.copy()
    
    
    for product in cleanedData.values():
        # Clean title
        title = product['title']
        cleanedTitle = cleanString(title)
        product['title'] = cleanedTitle
       
        # Clean features
        for feature in product['featuresMap']:
            value = str(product['featuresMap'][feature])
            cleanedValue = cleanString(value)
            product['featuresMap'][feature] = cleanedValue
    return cleanedData

def cleanString(value):
    cleanedValue = value
    inch_representations = ['Inch', 'inches', '"', '-inch', ' inch', 'inch']
    hertz_representations = ['Hertz', 'hertz', 'Hz', 'HZ', ' hz', '-hz', 'hz']
    pounds_representations = ['lbs', ' lb', 'pounds', 'Pounds', 'Lb', 'Lbs', 'lbs.', 'lb.']

    # Replace various representations with standardized terms
    for rep in inch_representations:
        if rep in value:
            cleanedValue = cleanedValue.replace(rep, 'inch')
    for rep in hertz_representations:
        if rep in value:
            cleanedValue = cleanedValue.replace(rep, 'hz')
    for rep in pounds_representations:
        if rep in value:
            cleanedValue = cleanedValue.replace(rep, 'lb')

    return cleanedValue
    