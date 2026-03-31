# Error Analysis

- **All headlines wrong:** 13
- **Only S1 correct:** 1 (retrieval-critical)
- **Only S2+R correct:** 4 (supervised advantage)
- **Only S3+R correct:** 0 (CLM advantage)

## Top-5 failures: S1

**Q:** On what date did Bond commence these proceedings?
- Type: date, Gold: 2023-03-09
- System: 1993-03-09, Malformed: False

**Q:** On what date was the order with reasons issued?
- Type: date, Gold: 2026-01-21
- System: 2025-07-22, Malformed: False

**Q:** Name the two private-rule mechanisms that the two statutes treat as relevant to transaction processing or validity: one 
- Type: names, Gold: ['Approved Form', 'Business Rules of an Authorised Market Institution']
- System: ['Business Rules', 'endorsement or instruction'], Malformed: False

**Q:** What legal principle did the Claimant rely on in support of the proposed cross-appeal?
- Type: name, Gold: The Crabtree principle
- System: Crabtree principle, Malformed: False

**Q:** How many Parts are listed in the contents before Schedule 1?
- Type: number, Gold: 8
- System: 4, Malformed: False

## Top-5 failures: S2+R

**Q:** Which two parties can a Reporting Financial Institution rely on for self-certification under Section IX B?
- Type: names, Gold: ['Account Holder', 'Controlling Person']
- System: ['Account Holder', 'Controlling Person of a Passive NFE'], Malformed: False

**Q:** On what date was the order with reasons issued?
- Type: date, Gold: 2026-01-21
- System: 2025-03-19, Malformed: False

**Q:** Name the two private-rule mechanisms that the two statutes treat as relevant to transaction processing or validity: one 
- Type: names, Gold: ['Approved Form', 'Business Rules of an Authorised Market Institution']
- System: ['Specific performance', 'Transfer of title to property'], Malformed: False

**Q:** What legal principle did the Claimant rely on in support of the proposed cross-appeal?
- Type: name, Gold: The Crabtree principle
- System: Crabtree principle, Malformed: False

**Q:** How many Parts are listed in the contents before Schedule 1?
- Type: number, Gold: 8
- System: 4, Malformed: False

## Top-5 failures: S3+R

**Q:** On what date was the order with reasons issued?
- Type: date, Gold: 2026-01-21
- System: 2025-08-08, Malformed: False

**Q:** Name the two private-rule mechanisms that the two statutes treat as relevant to transaction processing or validity: one 
- Type: names, Gold: ['Approved Form', 'Business Rules of an Authorised Market Institution']
- System: ['Authorised Market Institution', 'Authorised person'], Malformed: False

**Q:** What legal principle did the Claimant rely on in support of the proposed cross-appeal?
- Type: name, Gold: The Crabtree principle
- System: Crabtree principle, Malformed: False

**Q:** How many Parts are listed in the contents before Schedule 1?
- Type: number, Gold: 8
- System: 4, Malformed: False

**Q:** What was the name of the agreement under which the Claimant provided marketing and branding services?
- Type: name, Gold: 10-year Partnership and Services Agreement
- System: Partnership and Services Agreement, Malformed: False
