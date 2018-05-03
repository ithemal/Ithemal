-- USE @db_name;

ALTER TABLE code
      ADD num_instr int(12),
      ADD span int(12);

ALTER TABLE times
      ADD count int(12);
