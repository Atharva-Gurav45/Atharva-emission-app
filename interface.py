from flask import Flask, request, jsonify, render_template
from project_app.utils import EmissionClassifier
import config

app = Flask(__name__)

NUM_FIELDS = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'Year']


def read_input(data):
    """Convert incoming data (form / json / query) into correct types."""
    values = {}
    for f in NUM_FIELDS:
        values[f] = float(data[f])
    values['Year'] = int(values['Year'])
    return values


###################################################################################################################
################################################## Homepage API ###################################################
###################################################################################################################

@app.route('/')
def emission_homepage():
    print('Welcome to the Gas Turbine Emission Prediction Homepage')
    return render_template('index.html')


###################################################################################################################
################################################## Model API ######################################################
###################################################################################################################

@app.route('/predict', methods=['GET', 'POST'])
def get_emission_class():
    try:
        if request.method == 'POST':
            print('We are using POST method')
            # json (from the web page / postman) or form data
            data = request.get_json(silent=True) or request.form
        else:
            print('We are using GET method')
            data = request.args

        print('Data:', data)
        values = read_input(data)

        turbine = EmissionClassifier(**values)
        label = str(turbine.get_predicted_class())

        return jsonify({
            'prediction': label,
            'Result': f'Predicted emission class for the given turbine data is: {label}'
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=True)
