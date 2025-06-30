from flask import Flask, render_template, request, jsonify
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, CategoriesOptions, SentimentOptions
import os
from dotenv import load_dotenv
import logging
from cloudant.client import Cloudant
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def validate_config():
    required_vars = ['NLU_APIKEY', 'NLU_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    logger.info("Configuration validated successfully")

def initialize_watson_nlu():
    try:
        validate_config()
        authenticator = IAMAuthenticator(os.getenv('NLU_APIKEY'))
        nlu = NaturalLanguageUnderstandingV1(
            version='2023-03-25',
            authenticator=authenticator
        )
        nlu.set_service_url(os.getenv('NLU_URL'))
        logger.info("Watson NLU initialized successfully")
        return nlu
    except Exception as e:
        logger.error(f"Failed to initialize Watson NLU: {str(e)}")
        raise

def initialize_cloudant():
    try:
        cloudant_user = os.getenv('CLOUDANT_USERNAME')
        cloudant_api_key = os.getenv('CLOUDANT_APIKEY')
        cloudant_url = os.getenv('CLOUDANT_URL')
        db_name = os.getenv('CLOUDANT_DB')

        logger.info("Cloudant config check:")
        logger.info(f"  Username: {'SET' if cloudant_user else 'NOT SET'}")
        logger.info(f"  API Key: {'SET' if cloudant_api_key else 'NOT SET'}")
        logger.info(f"  URL: {'SET' if cloudant_url else 'NOT SET'}")
        logger.info(f"  DB Name: {db_name if db_name else 'NOT SET'}")

        if not all([cloudant_user, cloudant_api_key, cloudant_url, db_name]):
            logger.warning("Missing Cloudant credentials - database storage disabled")
            return None, None

        client = Cloudant.iam(cloudant_user, cloudant_api_key, connect=True, url=cloudant_url)
        session = client.session()
        logger.info(f"Cloudant session established: {session}")

        if db_name in client.all_dbs():
            db = client[db_name]
            logger.info(f"Connected to existing database: {db_name}")
        else:
            db = client.create_database(db_name)
            logger.info(f"Created new database: {db_name}")

        return client, db
    except Exception as e:
        logger.error(f"Cloudant initialization failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return None, None

# Initialize services
try:
    nlu = initialize_watson_nlu()
except Exception as e:
    logger.error(f"Application startup failed: {str(e)}")
    nlu = None

cloudant_client, cloudant_db = initialize_cloudant()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    status = {
        'nlu': 'healthy' if nlu else 'unavailable',
        'cloudant': 'healthy' if cloudant_db is not None else 'unavailable'
    }
    if nlu is None:
        return jsonify({'status': 'error', 'message': 'Watson NLU not initialized', 'services': status}), 503
    return jsonify({'status': 'healthy', 'service': 'news-classifier', 'services': status})

@app.route('/test-cloudant', methods=['POST'])
def test_cloudant():
    if cloudant_db is not None:
        try:
            test_doc = {
                'test': True,
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'This is a test document'
            }
            result = cloudant_db.create_document(test_doc)
            logger.info(f"Test document created: {result}")
            all_docs = cloudant_db.all_docs()
            doc_count = len(all_docs['rows'])
            return jsonify({
                'success': True,
                'document_id': result['_id'],
                'total_documents': doc_count,
                'message': 'Test document created successfully'
            })
        except Exception as e:
            logger.error(f"Cloudant test failed: {str(e)}")
            return jsonify({'error': f'Cloudant test failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Cloudant not initialized'}), 503

@app.route('/analyze', methods=['POST'])
def analyze():
    if nlu is None:
        return jsonify({'error': 'Watson NLU service not available. Check your configuration.'}), 503

    try:
        text = request.form.get('news', '').strip()
        if not text:
            return jsonify({'error': 'No text provided for analysis'}), 400
        if len(text) < 15:
            return jsonify({'error': 'Text too short for meaningful analysis (minimum 15 characters)'}), 400
        if len(text) > 50000:
            text = text[:50000]
            logger.warning("Text truncated to 50,000 characters")

        logger.info(f"Analyzing text of length: {len(text)}")

        response = nlu.analyze(
            text=text,
            features=Features(
                sentiment=SentimentOptions(),
                categories=CategoriesOptions(limit=3),
                keywords=KeywordsOptions(limit=5)
            )
        ).get_result()

        logger.info("Analysis completed successfully")

        if cloudant_db is not None:
            try:
                document = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "input_text": text[:1000],
                    "input_text_length": len(text),
                    "analysis_result": response
                }
                logger.info("Attempting to store document in Cloudant...")
                result = cloudant_db.create_document(document)
                if result.exists():
                    logger.info(f"Document successfully stored with ID: {result['_id']}")
                else:
                    logger.error("Document creation returned success but document doesn't exist")
            except Exception as ce:
                logger.error(f"Failed to store analysis in Cloudant: {str(ce)}")
                logger.error(f"Error type: {type(ce).__name__}")
        else:
            logger.warning("Cloudant database not available - skipping storage")

        return jsonify(response)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Analysis failed: {error_msg}")
        if "unauthorized" in error_msg.lower():
            return jsonify({'error': 'Invalid API credentials. Check your Watson NLU API key.'}), 401
        elif "not enough text" in error_msg.lower():
            return jsonify({'error': 'Not enough text for analysis. Please provide more content.'}), 400
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return jsonify({'error': 'API usage limit reached. Try again later.'}), 429
        else:
            return jsonify({'error': f'Analysis failed: {error_msg}'}), 500

@app.route('/db-status')
def db_status():
    if cloudant_db is not None:
        try:
            doc_count = len(cloudant_db)
            recent_docs = [doc['_id'] for doc in cloudant_db if '_id' in doc][-5:]
            return jsonify({
                'database_name': cloudant_db.database_name,
                'document_count': doc_count,
                'recent_documents': recent_docs
            })
        except Exception as e:
            logger.error(f"Failed to get database status: {str(e)}")
            return jsonify({'error': f'Database status check failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Cloudant not initialized'}), 503

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if nlu is None:
        print("ERROR: Cannot start application - Watson NLU initialization failed")
        exit(1)

    print("Starting News Classifier App...")
    print(f"Cloudant status: {'Connected' if cloudant_db is not None else 'Not available'}")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
