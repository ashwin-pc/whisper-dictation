#!/usr/bin/env python3
import boto3
import json
import os
from dotenv import load_dotenv
from logger_config import setup_logging
from text_safety import build_bedrock_request, sanitize_for_typing

# Load environment variables
load_dotenv()
logger = setup_logging()

class BedrockClient:
    def __init__(self):
        """Initialize Bedrock runtime client"""
        self.region_name = os.getenv('AWS_REGION', 'us-east-1')
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
        
        # Initialize boto3 client for Bedrock Runtime
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.region_name,
            )
            logger.info(f"Bedrock client initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Error initializing Bedrock client: {e}")
            self.client = None
    
    def enhance_text(self, transcribed_text, selected_text):
        """
        Send transcribed instruction and selected text to Claude via Bedrock.
        Returns the enhanced/modified text.
        """
        if not self.client:
            raise Exception("Bedrock client not initialized")
        
        # Keep trusted operating instructions in the system field and encode
        # selected/transcribed text as bounded JSON data in the user message.
        request_body = build_bedrock_request(transcribed_text, selected_text)
        
        try:
            # Invoke the model
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            # Extract the generated text
            if 'content' in response_body and len(response_body['content']) > 0:
                enhanced_text = sanitize_for_typing(
                    response_body['content'][0]['text']
                )
                if not enhanced_text:
                    raise ValueError("Bedrock returned no safe text to type")
                logger.debug(f"Bedrock response: {enhanced_text}")
                return enhanced_text
            else:
                raise Exception("No content in Bedrock response")
                
        except Exception as e:
            logger.error(f"Error calling Bedrock: {e}")
            raise
    
    def test_connection(self):
        """Test if Bedrock client is working properly"""
        if not self.client:
            return False, "Client not initialized"
        
        try:
            # Simple test with a basic prompt
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'Hello, Bedrock connection successful!'"
                    }
                ]
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            if 'content' in response_body and len(response_body['content']) > 0:
                return True, response_body['content'][0]['text'].strip()
            else:
                return False, "No content in response"
                
        except Exception as e:
            return False, str(e)
    
    def is_available(self):
        """Check if Bedrock client is available and configured"""
        return self.client is not None
    
    def get_model_info(self):
        """Get information about the current model configuration"""
        return {
            'model_id': self.model_id,
            'region': self.region_name,
            'available': self.is_available()
        }
