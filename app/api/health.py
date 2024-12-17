import os
from flask import jsonify, Blueprint


health_blueprint = Blueprint('health', __name__)

@health_blueprint.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Libbi is Live"})