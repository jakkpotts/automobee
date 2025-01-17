-- Rename metadata columns to extra_data to avoid SQLAlchemy reserved keyword
ALTER TABLE detections ADD COLUMN extra_data JSONB;
ALTER TABLE target_matches ADD COLUMN extra_data JSONB;
ALTER TABLE system_metrics ADD COLUMN extra_data JSONB;
ALTER TABLE performance_logs ADD COLUMN extra_data JSONB;

-- Copy data from old columns if they exist
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'detections' AND column_name = 'metadata') THEN
        UPDATE detections SET extra_data = metadata;
        ALTER TABLE detections DROP COLUMN metadata;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'target_matches' AND column_name = 'metadata') THEN
        UPDATE target_matches SET extra_data = metadata;
        ALTER TABLE target_matches DROP COLUMN metadata;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'system_metrics' AND column_name = 'metadata') THEN
        UPDATE system_metrics SET extra_data = metadata;
        ALTER TABLE system_metrics DROP COLUMN metadata;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_logs' AND column_name = 'metadata') THEN
        UPDATE performance_logs SET extra_data = metadata;
        ALTER TABLE performance_logs DROP COLUMN metadata;
    END IF;
END $$; 