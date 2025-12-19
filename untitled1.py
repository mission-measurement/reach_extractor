import re
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
import warnings
import argparse

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class ReachInfo:
    has_reach: bool = False
    reach_number: Optional[int] = None
    reach_number_raw: Optional[str] = None
    reach_unit: Optional[str] = None
    reach_snippet: Optional[str] = None
    confidence_score: float = 0.0
    extraction_method: str = "none"
    warnings: List[str] = field(default_factory=list)


class ReachExtractorV5:

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.beneficiary_words = {
            'member', 'members', 'participant', 'participants', 'attendee', 'attendees',
            'student', 'students', 'scholar', 'scholars', 'learner', 'learners',
            'family', 'families', 'household', 'households',
            'individual', 'individuals', 'people', 'person', 'persons',
            'beneficiary', 'beneficiaries', 'recipient', 'recipients',
            'client', 'clients', 'customer', 'customers',
            'trainee', 'trainees',
            'collegian', 'collegians',
            'apprentice', 'apprentices',
            'fellow', 'fellows',
            'enrollee', 'enrollees',
            'mentee', 'mentees',
            'graduate', 'graduates',
            'intern', 'interns',
            'professional', 'professionals', 'worker', 'workers',
            'employee', 'employees',
            'youth', 'children', 'child', 'kid', 'kids', 'teen', 'teens', 'teenager', 'teenagers',
            'adult', 'adults', 'senior', 'seniors', 'elder', 'elders',
            'veteran', 'veterans', 'refugee', 'refugees', 'immigrant', 'immigrants',
            'resident', 'residents', 'citizen', 'citizens',
            'patient', 'patients', 'visitor', 'visitors', 'guest', 'guests',
            'volunteer', 'volunteers',
            'teacher', 'teachers', 'educator', 'educators',
            'farmer', 'farmers',
            'woman', 'women', 'man', 'men', 'girl', 'girls', 'boy', 'boys',
            'home', 'homes',
        }

        self.non_beneficiary_nouns = {
            'school', 'schools', 'site', 'sites', 'location', 'locations',
            'center', 'centers', 'clinic', 'clinics', 'office', 'offices',
            'route', 'routes',
            'county', 'counties', 'state', 'states', 'city', 'cities',
            'district', 'districts', 'region', 'regions', 'area', 'areas',
            'country', 'countries', 'nation', 'nations', 'community', 'communities',
            'week', 'weeks', 'day', 'days', 'hour', 'hours', 'month', 'months',
            'year', 'years', 'minute', 'minutes', 'session', 'sessions',
            'visit', 'visits',
            'program', 'programs', 'project', 'projects', 'event', 'events',
            'class', 'classes', 'course', 'courses',
            'procedure', 'procedures', 'service', 'services',
            'outing', 'outings', 'trip', 'trips',
            'meal', 'meals', 'book', 'books', 'item', 'items',
            'grant', 'grants', 'dollar', 'dollars',
            'company', 'companies', 'business', 'businesses', 'organization', 'organizations',
            'smb', 'smbs', 'fintech', 'fintechs', 'network', 'networks',
        }

        self._build_patterns()
        print("✅ Reach Extractor v5 ready!")

    def _log(self, msg: str):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def _build_patterns(self):
        beneficiaries = '|'.join(sorted(self.beneficiary_words, key=len, reverse=True))
        num = r'(\d+(?:,\d{3})*\+?)'
        quant = r'(?:over\s+|more\s+than\s+|approximately\s+|approx\.?\s+|about\s+|nearly\s+|up\s+to\s+|at\s+least\s+|~\s*)?'
        gap = r'(?:\s+(?!schools?\b|sites?\b|locations?\b|weeks?\b|days?\b|hours?\b|months?\b|percent|%|visits?\b|routes?\b)\S+){0,5}?\s+'

        self.patterns = [
            (re.compile(rf'(?:serves?|served|serving)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.95, 'serve'),
            (re.compile(rf'(?:helps?|helped|helping)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.95, 'help'),
            (re.compile(rf'(?:supports?|supported)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.95, 'support'),
            (re.compile(rf'(?:reaches?|reached|reaching)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.90, 'reach'),
            (re.compile(rf'(?:impacts?|impacted)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.90, 'impact'),
            (re.compile(rf'(?:educates?|educated|educating)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.90, 'educate'),
            (re.compile(rf'(?:trains?|trained|training)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.90, 'train'),
            (re.compile(rf'(?:teaches?|taught|teaching)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.90, 'teach'),
            (re.compile(rf'(?:enrolls?|enrolled|enrolling)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.90, 'enroll'),
            (re.compile(rf'(?:program|service|care)\s+(?:for|to)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.85, 'program_for'),
            (re.compile(rf'(?:provides?|provided|providing)\s+(?:\w+\s+){0,3}?(?:for|to)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.85, 'provide_for'),
            (re.compile(rf'(?:delivers?|delivered)\s+(?:\w+\s+){0,3}?(?:for|to)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.85, 'deliver_to'),
            (re.compile(rf'(?:served?\s+)?(?:in\s+\d+\s+schools?\s+)?with\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.80, 'with'),
            (re.compile(rf'{num}{gap}({beneficiaries})\s+(?:annually|yearly|per\s+year|a\s+year|each\s+year)', re.I), 0.90, 'annual'),
            (re.compile(rf'(?:annually|yearly)\s+(?:serves?|serving|reaching)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.90, 'annual_verb'),
            (re.compile(rf'(?:has|have)\s+(?:served|helped|reached|supported|trained)\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.85, 'has_served'),
            (re.compile(rf'{quant}{num}{gap}({beneficiaries})\s+(?:have|has)\s+graduated', re.I), 0.85, 'have_graduated'),
            (re.compile(rf'(?:more\s+than|over)\s+{num}{gap}({beneficiaries})\s+(?:have|has)\s+graduated', re.I), 0.90, 'have_graduated'),
            (re.compile(rf'(?:home|center|clinic|provider|resource)\s+for\s+{quant}{num}{gap}({beneficiaries})\b', re.I), 0.80, 'facility_for'),
            (re.compile(rf'visits?\s+to\s+(?:approximately\s+|about\s+|over\s+)?{num}\s+({beneficiaries})\b', re.I), 0.90, 'visits_to'),
            (re.compile(rf'pounds?\s+(?:of\s+\w+\s+)?to\s+(?:over\s+|approximately\s+|about\s+)?{num}\s+({beneficiaries})\b', re.I), 0.92, 'pounds_to'),
            (re.compile(rf'\b{num}{gap}({beneficiaries})\b', re.I), 0.60, 'standalone'),
        ]

        beneficiary_list = r'(?:adults?|children|kids?|men|women|boys?|girls?|families|individuals?|people|patients?|clients?|students?|members?|participants?|youth|seniors?|veterans?|attendees?|mentors?)'
        self.x_and_y_pattern = re.compile(
            rf'\b{num}\s+{beneficiary_list}\s+and\s+{num}\s+{beneficiary_list}\b',
            re.I
        )

    def _parse_number(self, num_str: str) -> Optional[int]:
        try:
            cleaned = num_str.replace(',', '').replace('~', '').replace('+', '').strip()
            return int(cleaned)
        except:
            return None

    def _is_false_positive(self, text: str, match_start: int, match_end: int, number: int, unit: str) -> Tuple[bool, str]:
        context_start = max(0, match_start - 60)
        context_end = min(len(text), match_end + 60)
        context = text[context_start:context_end]
        context_lower = context.lower()
        text_lower = text.lower()

        if re.search(rf'9\s*/\s*{number}\b', context) or re.search(rf'9/{number}\b', context):
            return True, "9/11_pattern"
        if re.search(rf'\b9-{number}\b', context):
            return True, "9/11_pattern"
        if re.search(rf'post-9/{number}', context, re.I):
            return True, "9/11_pattern"

        if re.search(rf'between\s+{number}\s+and\s+\d+', context_lower):
            return True, "between_X_and_Y"
        if re.search(rf'between\s+\d+\s+and\s+{number}', context_lower):
            return True, "between_X_and_Y"
        if re.search(rf'\b{number}\s+(?:to|and)\s+\d+\s+years?\s+old', context_lower):
            return True, "X_to_Y_years_old"
        if re.search(rf'\b\d+\s+(?:to|and)\s+{number}\s+years?\s+old', context_lower):
            return True, "X_to_Y_years_old"

        if re.search(rf'\b{number}\s+routes?\b', context_lower):
            return True, "routes_not_people"
        if re.search(rf'\b{number}\s+sites?\b', context_lower):
            return True, "sites_not_people"
        if re.search(rf'\b{number}\s+locations?\b', context_lower):
            return True, "locations_not_people"

        if re.search(rf'Act\s+of\s+{number}', context, re.I):
            return True, "legislation_year"
        if re.search(rf'Section\s+\d+\w?\s+of\s+.*{number}', context, re.I):
            return True, "legislation_year"
        if 1900 <= number <= 2099:
            if re.search(rf'\bof\s+{number}\b', context_lower):
                return True, "year_reference"

        if re.search(rf'pre-?k\s*[-–—]\s*{number}', context_lower):
            return True, "prek_grade_range"
        if re.search(rf'prek\s*[-–—]\s*{number}', context_lower):
            return True, "prek_grade_range"
        if re.search(rf'\bk\s*[-–—]\s*{number}\b', context_lower):
            return True, "k_grade_range"

        if unit.lower() in ['visit', 'visits']:
            return True, "visits_not_people"

        if re.search(rf'\b{number}\s*%', context):
            return True, "percentage"
        if re.search(rf'\b{number}\s+percent', context_lower):
            return True, "percentage"

        if re.search(rf'ages?\s+{number}\s*[-–—to]+\s*\d+', context_lower):
            return True, "age_range"
        if re.search(rf'ages?\s+\d+\s*[-–—to]+\s*{number}', context_lower):
            return True, "age_range"
        if re.search(rf'aged\s+{number}', context_lower):
            return True, "age_range"
        if re.search(rf'\({number}\s*[-–—]\s*\d+\)', context):
            return True, "age_range_parens"
        if re.search(rf'\(\d+\s*[-–—]\s*{number}\)', context):
            return True, "age_range_parens"
        if re.search(rf'\d+\s*[-–—]\s*{number}\s+and\s+(?:their|the)', context_lower):
            return True, "age_range_end"
        if re.search(rf'\d{{1,2}}\s*[-–—]\s*{number}\s+and\b', context_lower):
            if number <= 99:
                return True, "age_range_end"

        if re.search(rf'grades?\s+{number}\s*[-–—to]+\s*\d+', context_lower):
            return True, "grade_range"
        if re.search(rf'grades?\s+\d+\s*[-–—to]+\s*{number}', context_lower):
            return True, "grade_range"
        if re.search(rf'\b{number}(?:st|nd|rd|th)\s+grade', context_lower):
            return True, "specific_grade"

        if re.search(rf'\b{number}\s+weeks?\b', context_lower):
            return True, "duration_weeks"
        if re.search(rf'\b{number}\s+days?\b', context_lower):
            return True, "duration_days"
        if re.search(rf'\b{number}\s+hours?\b', context_lower):
            return True, "duration_hours"
        if re.search(rf'\b{number}\s+months?\b', context_lower):
            return True, "duration_months"
        if re.search(rf'\b{number}\s+minutes?\b', context_lower):
            return True, "duration_minutes"
        if re.search(rf'\b{number}\s+sessions?\b', context_lower):
            return True, "duration_sessions"

        if 1900 <= number <= 2030:
            year_indicators = [
                rf'\bin\s+{number}\b',
                rf'\bsince\s+{number}\b',
                rf'\bfrom\s+{number}\b',
                rf'\bfounded\s+(?:in\s+)?{number}\b',
                rf'\bestablished\s+(?:in\s+)?{number}\b',
                rf'\blaunched\s+(?:in\s+)?{number}\b',
                rf'\bstarted\s+(?:in\s+)?{number}\b',
                rf'\bopened\s+(?:in\s+)?{number}\b',
                rf'\b{number}\s*[-–—]\s*(?:19|20)?\d{{2}}',
                rf'(?:19|20)\d{{2}}\s*[-–—]\s*{number}',
                rf'\bfy\s*{number}',
                rf'\bfiscal\s+(?:year\s+)?{number}',
            ]
            for pattern in year_indicators:
                if re.search(pattern, context_lower):
                    return True, "year_reference"

        if re.search(rf'from\s+{number}\s+schools?\b', context_lower):
            return True, "school_count"
        if re.search(rf'{number}\s+schools?\s+(?:all\s+)?over\b', context_lower):
            return True, "school_count"
        if re.search(rf'through\s+(?:over\s+)?{number}\s+(?:smbs?|companies|businesses)', context_lower):
            return True, "business_count"

        num_formatted = f"{number:,}"
        if re.search(rf'\$\s*{num_formatted}\b', text) or re.search(rf'\$\s*{number}\b', text):
            return True, "dollar_amount"
        if re.search(rf'\$\s*{num_formatted}\b', context) or re.search(rf'\$\s*{number}\b', context):
            return True, "dollar_amount"

        if re.search(rf'\b{num_formatted}\s*pounds?\b', text, re.I) or re.search(rf'\b{number}\s*pounds?\b', text, re.I):
            if not re.search(rf'pounds?\s+(?:of\s+\w+\s+)?to\s+(?:over\s+|approximately\s+|about\s+)?{num_formatted}', text, re.I):
                if not re.search(rf'pounds?\s+(?:of\s+\w+\s+)?to\s+(?:over\s+|approximately\s+|about\s+)?{number}', text, re.I):
                    return True, "pounds_weight"

        if number == 50 and re.search(r'50\s*/\s*50', text):
            return True, "fifty_fifty_raffle"

        if number == 500 and re.search(r'fortune\s*500', text_lower):
            return True, "fortune_500"

        if number == 501 and re.search(r'501\s*\(\s*c\s*\)', text, re.I):
            return True, "501c3_status"

        if re.search(rf'(?:section|title|irc|hud)\s+{number}\b', text_lower):
            return True, "section_number"

        if re.search(rf'\b{number}\s+patient\s+days?\b', text_lower):
            return True, "patient_days"

        if re.search(rf'\b{number}\s+(?:assessments?|procedures?|nights?)\b', text_lower):
            return True, "non_people_count"

        if re.search(rf'\bpk\s*[-–—]\s*{number}\b', context_lower):
            return True, "pk_grade_range"

        if re.search(rf'{number}\s+dollars?\b', context_lower):
            return True, "money"

        if re.search(rf'{number}(?:st|nd|rd|th)\s+century', context_lower):
            return True, "century"

        return False, ""

    def _get_snippet(self, text: str, start: int, end: int) -> str:
        snippet_start = max(0, start - 40)
        snippet_end = min(len(text), end + 40)
        snippet = text[snippet_start:snippet_end].strip()
        snippet = re.sub(r'\s+', ' ', snippet)
        if snippet_start > 0:
            snippet = '...' + snippet
        if snippet_end < len(text):
            snippet = snippet + '...'
        return snippet

    def _standardize_unit(self, unit: str) -> str:
        unit = unit.lower().strip()
        mappings = {
            'member': 'members', 'participant': 'participants',
            'student': 'students', 'scholar': 'scholars', 'learner': 'learners',
            'family': 'families', 'household': 'households',
            'individual': 'individuals', 'person': 'people', 'persons': 'people',
            'child': 'children', 'kid': 'children', 'kids': 'children',
            'youth': 'youth', 'teen': 'teens',
            'adult': 'adults', 'senior': 'seniors',
            'patient': 'patients', 'client': 'clients',
            'veteran': 'veterans', 'refugee': 'refugees',
            'volunteer': 'volunteers', 'teacher': 'teachers',
            'woman': 'women', 'man': 'men', 'girl': 'girls', 'boy': 'boys',
            'home': 'homes', 'resident': 'residents',
            'trainee': 'trainees', 'collegian': 'collegians',
            'apprentice': 'apprentices', 'fellow': 'fellows',
            'enrollee': 'enrollees', 'mentee': 'mentees',
            'graduate': 'graduates', 'intern': 'interns',
        }
        return mappings.get(unit, unit)

    def extract_reach(self, text: str) -> ReachInfo:
        if not text or pd.isna(text):
            return ReachInfo(extraction_method="no_input")

        text = str(text).strip()
        if len(text) < 10:
            return ReachInfo(extraction_method="text_too_short")

        candidates = []
        warnings_list = []

        x_and_y_matches = list(self.x_and_y_pattern.finditer(text))
        summed_counts = {}

        for match in x_and_y_matches:
            groups = match.groups()
            if len(groups) >= 2:
                num1 = self._parse_number(groups[0])
                num2 = self._parse_number(groups[1])
                if num1 and num2:
                    total = num1 + num2
                    summed_counts[num1] = total
                    summed_counts[num2] = total

                    snippet = self._get_snippet(text, match.start(), match.end())
                    candidates.append({
                        'number': total,
                        'number_raw': f"{num1}+{num2}",
                        'unit': 'people',
                        'confidence': 0.88,
                        'method': 'x_and_y_sum',
                        'snippet': snippet,
                    })
                    self._log(f"Found X+Y pattern: {num1} + {num2} = {total}")

        for pattern, base_confidence, method in self.patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 2:
                    num_str, unit = groups[0], groups[1]
                    number = self._parse_number(num_str)

                    if not number or number < 10:
                        continue

                    if number in summed_counts:
                        self._log(f"Skipping {number} - part of X+Y pair summed to {summed_counts[number]}")
                        continue

                    is_fp, fp_reason = self._is_false_positive(
                        text, match.start(), match.end(), number, unit
                    )

                    if is_fp:
                        self._log(f"Filtered false positive: {number} {unit} ({fp_reason})")
                        continue

                    unit_lower = unit.lower()
                    if unit_lower not in self.beneficiary_words:
                        self._log(f"Unit not in beneficiary list: {unit_lower}")
                        continue

                    confidence = base_confidence

                    if number >= 100:
                        confidence += 0.03
                    if number >= 1000:
                        confidence += 0.02

                    confidence = min(0.99, confidence)

                    snippet = self._get_snippet(text, match.start(), match.end())

                    candidates.append({
                        'number': number,
                        'number_raw': num_str,
                        'unit': self._standardize_unit(unit_lower),
                        'confidence': confidence,
                        'method': method,
                        'snippet': snippet,
                    })

        if candidates:
            candidates.sort(key=lambda x: (x['confidence'], x['number']), reverse=True)

            seen = set()
            unique = []
            for c in candidates:
                if c['number'] not in seen:
                    seen.add(c['number'])
                    unique.append(c)

            best = unique[0]

            if len(unique) > 1:
                warnings_list.append(f"Multiple candidates: {len(unique)}")

            return ReachInfo(
                has_reach=True,
                reach_number=best['number'],
                reach_number_raw=best['number_raw'],
                reach_unit=best['unit'],
                reach_snippet=best['snippet'],
                confidence_score=best['confidence'],
                extraction_method=best['method'],
                warnings=warnings_list
            )

        return ReachInfo(
            has_reach=False,
            extraction_method="no_match",
            warnings=["No valid reach patterns matched"]
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        description_column: str = 'programdescription',
        id_column: str = '_id',
        show_progress: bool = True
    ) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame")

        if description_column not in df.columns:
            raise ValueError(f"Column '{description_column}' not found")

        results = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Extracting reach") if show_progress else df.iterrows()

        for idx, row in iterator:
            try:
                program_id = row.get(id_column, idx) if id_column in df.columns else idx
                description = row.get(description_column, '')
                reach = self.extract_reach(description)

                results.append({
                    id_column: program_id,
                    'has_reach': reach.has_reach,
                    'reach_number': reach.reach_number,
                    'reach_number_raw': reach.reach_number_raw,
                    'reach_unit': reach.reach_unit,
                    'reach_snippet': reach.reach_snippet,
                    'confidence_score': reach.confidence_score,
                    'extraction_method': reach.extraction_method,
                    'warnings': '; '.join(reach.warnings) if reach.warnings else None,
                    description_column: description
                })
            except Exception as e:
                results.append({
                    id_column: row.get(id_column, idx),
                    'has_reach': False,
                    'reach_number': None,
                    'reach_number_raw': None,
                    'reach_unit': None,
                    'reach_snippet': None,
                    'confidence_score': 0.0,
                    'extraction_method': 'error',
                    'warnings': str(e),
                    description_column: row.get(description_column, '')
                })

        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Reach Extractor v5')
    parser.add_argument('programs_file', help='CSV file')
    parser.add_argument('--output', '-o', default='reach_results_v5.csv')
    parser.add_argument('--description-column', default='programdescription')
    parser.add_argument('--id-column', default='_id')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    print(f"Loading {args.programs_file}...")
    df = pd.read_csv(args.programs_file, encoding='utf-8-sig')
    print(f"✅ Loaded {len(df)} programs")

    extractor = ReachExtractorV5(debug=args.debug)
    results = extractor.process_dataframe(
        df,
        description_column=args.description_column,
        id_column=args.id_column
    )

    with_reach = results['has_reach'].sum()
    total = len(results)
    print(f"\n📊 Programs with reach: {with_reach}/{total} ({with_reach/total*100:.1f}%)")

    if with_reach > 0:
        print(f"Average confidence: {results[results['has_reach']]['confidence_score'].mean():.2f}")
        print("\nExtraction methods:")
        print(results[results['has_reach']]['extraction_method'].value_counts())

    results.to_csv(args.output, index=False)
    print(f"\n✅ Saved to {args.output}")


if __name__ == '__main__':
    main()
