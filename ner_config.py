"""Predeclared policy; model prompts are not a claim of label support."""

POLICY = '''first_name last_name full_name date_of_birth age gender race_ethnicity sexuality religion language nationality country city state county street_address postal_code phone_number email url ip_address mac_address username password company_name occupation employment_status salary income date time ssn passport_number national_id driver_license medical_record_number blood_type medical_condition medication health_insurance_number bank_account_number bank_routing_number swift_bic iban credit_debit_card credit_card_pin vehicle_identifier license_plate biometric other_pii education_level religious_belief biometric_identifier tax_id date_time pin health_plan_beneficiary_number'''.split()
ALIASES = {'full_name': 'name', 'postal_code': 'postcode', 'username': 'user_name',
           'credit_debit_card': 'credit_card_number', 'bank_account_number': 'account_number',
           'religion': 'religious_belief', 'biometric': 'biometric_identifier',
           'health_insurance_number': 'health_plan_beneficiary_number', 'credit_card_pin': 'pin'}
LABEL_MAP = {ALIASES.get(label, label): ALIASES.get(label, label) if label in
             {'religion', 'biometric', 'health_insurance_number', 'credit_card_pin'} else label
             for label in POLICY}
LABEL_MAP.pop('ip_address')
LABEL_MAP.update(ipv4='ip_address', ipv6='ip_address')
