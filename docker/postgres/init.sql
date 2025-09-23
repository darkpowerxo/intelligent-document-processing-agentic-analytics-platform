-- Initialize database schemas for AI Demo

-- MLflow schema (MLflow will create its own tables)
CREATE SCHEMA IF NOT EXISTS mlflow;

-- Application schema
CREATE SCHEMA IF NOT EXISTS app;

-- Create users table for authentication
CREATE TABLE IF NOT EXISTS app.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table for tracking processed documents
CREATE TABLE IF NOT EXISTS app.documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    original_name VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_size BIGINT NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    processed_at TIMESTAMP NULL,
    user_id INTEGER REFERENCES app.users(id),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create processing results table
CREATE TABLE IF NOT EXISTS app.processing_results (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES app.documents(id),
    processing_type VARCHAR(100) NOT NULL,
    result_data JSONB NOT NULL,
    confidence_score FLOAT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create agents table to track agent activities
CREATE TABLE IF NOT EXISTS app.agents (
    id SERIAL PRIMARY KEY,
    agent_type VARCHAR(100) NOT NULL,
    agent_name VARCHAR(200) NOT NULL,
    status VARCHAR(50) DEFAULT 'idle',
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_tasks_completed INTEGER DEFAULT 0,
    average_processing_time_ms INTEGER DEFAULT 0,
    configuration JSONB DEFAULT '{}'::jsonb
);

-- Create agent tasks table
CREATE TABLE IF NOT EXISTS app.agent_tasks (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES app.agents(id),
    task_type VARCHAR(100) NOT NULL,
    task_data JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    result JSONB NULL,
    error_message TEXT NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_status ON app.documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_upload_time ON app.documents(upload_time);
CREATE INDEX IF NOT EXISTS idx_processing_results_document ON app.processing_results(document_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON app.agent_tasks(status);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_created ON app.agent_tasks(created_at);

-- Insert default admin user (password: admin123 - hashed with bcrypt)
INSERT INTO app.users (username, email, hashed_password) VALUES 
('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj7.k.7Y.fG2')
ON CONFLICT (username) DO NOTHING;

-- Insert default agents
INSERT INTO app.agents (agent_type, agent_name, configuration) VALUES 
('document_analyzer', 'Document Analyzer Agent', '{"max_concurrent_tasks": 5, "timeout_seconds": 60}'),
('business_intelligence', 'Business Intelligence Agent', '{"analysis_depth": "comprehensive", "include_visualizations": true}'),
('quality_assurance', 'Quality Assurance Agent', '{"validation_rules": ["completeness", "accuracy", "consistency"]}'),
('orchestrator', 'Orchestration Agent', '{"max_parallel_workflows": 10, "priority_handling": true}')
ON CONFLICT DO NOTHING;