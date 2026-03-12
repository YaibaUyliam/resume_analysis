from __future__ import annotations

import json
import re
import os
import datetime
import hashlib
import unicodedata

from difflib import SequenceMatcher
from urllib.parse import urlparse


def convert_duration_to_dates(duration: str):
    """
    Convert duration string to startDate, endDate
    Rules:
        - yyyy -> yyyy
        - yyyy.mm -> yyyy-mm
        - yyyy.mm.dd -> yyyy-mm-dd
        - 'now' or '至今' -> now

    """
    if not duration:
        return None, None

    duration = (
        duration.replace("—", "-")
        .replace("–", "-")
        .replace("--", "-")
        .replace("——", "-")
        .replace("––", "-")
        .replace("~", "-")
        .replace(" ", "")
        .strip()
    )
    if "-" not in duration:
        return duration, None
    # start, end = [p.strip() for p in duration.split("-", 1)]
    parts = duration.split("-")
    if len(parts) > 2:
        start = "-".join(parts[:2])  
        end = "-".join(parts[2:])    
    else:
        start, end = [p.strip() for p in duration.split("-", 1)]
    # start = start.replace(".", "-")

    month_map = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "sep": "09",
        "sept": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }

    def normalize(part: str):
        p = part.strip().lower()
        if not p:
            return None

        # now / 至今
        if p in ["now", "Now", "至今", "现在"]:
            return "Now"

        # yyyy
        if re.match(r"^\d{4}$", p):
            return p

        # yyyy.mm or yyyy-mm or yyyy.mm or yyyy/mm
        if re.match(r"^\d{4}[.-/ ]\d{1,2}$", p):
            parts = list(filter(None, re.split(r"[.\-/ ]", p)))
            if len(parts) == 2:
                year, month = parts
            return f"{year}-{month.zfill(2)}"

        # yyyy.mm.dd or yyyy-mm-dd or yyyy/mm/dd or yyyy mm dd
        if re.match(r"^\d{4}[.-/ ]\d{1,2}[.-/ ]\d{1,2}$", p):
            year, month, day = re.split(r"[.-/ ]", p)
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Month Year ( ex: Feb 2022, July 2021)
        match = re.match(r"([A-Za-z]+)\s+(\d{4})", p)
        if match:
            mon, year = match.groups()
            mon = month_map.get(mon[:3].lower(), None)
            return f"{year}-{mon}" if mon else year

        # Chinese format: YYYY年M月 or YYYY年M月D日
        match = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})?日?", part)
        if match:
            year, month, day = match.groups()
            if day:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            return f"{year}-{month.zfill(2)}"

        return part  # fallback

    startDate = normalize(start)
    endDate = normalize(end)
    return startDate, endDate


def parse_years_of_experience(value: str):

    if not value or str(value).strip() == "":
        return None
    try:
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group(0))
        return None
    except ValueError:
        return None


def parse_float(value: str):
    if not value or str(value).strip() == "":
        return None
    try:
        digits = ''.join(filter(str.isdigit, value ))
        return float(digits)

    except ValueError:
        return None


def parse_array(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [pos.strip() for pos in value.split("/") if pos.strip()]
    return []


def normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def strip_accents(value: str | None) -> str:
    value = normalize_whitespace(value)
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: str | None) -> str:
    value = strip_accents(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return normalize_whitespace(value)


def normalize_email(value: str | None) -> str:
    value = normalize_whitespace(value).lower()
    if not value or "@" not in value:
        return ""
    return value


def normalize_phone(value: str | None) -> str:
    digits = re.sub(r"\D", "", value or "")
    if digits.startswith("84") and len(digits) > 9:
        digits = "0" + digits[2:]
    return digits


def normalize_profile_handle(value: str | None, domains: tuple[str, ...]) -> str:
    raw = normalize_whitespace(value).lower()
    if not raw:
        return ""

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc.lower().replace("www.", "")
    path_segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if host and any(domain in host for domain in domains):
        if "linkedin.com" in host and path_segments:
            return path_segments[-1].lower()
        if path_segments:
            return path_segments[0].lower()
        return ""

    if "/" not in raw and " " not in raw:
        return raw.strip("@")
    return ""


def normalize_linkedin(value: str | None) -> str:
    return normalize_profile_handle(value, ("linkedin.com",))


def normalize_github(value: str | None) -> str:
    return normalize_profile_handle(value, ("github.com",))


def normalize_name(value: str | None) -> str:
    return normalize_text(value)


def normalize_location(value: str | None) -> str:
    return normalize_text(value)


def similarity_ratio(value_a: str | None, value_b: str | None) -> float:
    norm_a = normalize_text(value_a)
    norm_b = normalize_text(value_b)
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def jaccard_similarity(values_a, values_b) -> float:
    set_a = {normalize_text(v) for v in values_a or [] if normalize_text(v)}
    set_b = {normalize_text(v) for v in values_b or [] if normalize_text(v)}
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def parse_skills(info: dict) -> list[str]:
    skills = []
    for skill in info.get("skills", []):
        if isinstance(skill, dict):
            skill_name = normalize_whitespace(skill.get("skill_name"))
            if skill_name:
                skills.append(skill_name)
    return skills


def get_latest_experience_item(info: dict) -> dict:
    for experience in info.get("experience", []):
        if isinstance(experience, dict) and (
            experience.get("company") or experience.get("position")
        ):
            return experience
    return {}


def get_latest_education_item(info: dict) -> dict:
    for education in info.get("education", []):
        if isinstance(education, dict) and (
            education.get("school_name") or education.get("major")
        ):
            return education
    return {}


def build_canonical_cv_text(info: dict) -> str:
    personal_info = info.get("personal_info", {})
    latest_experience = get_latest_experience_item(info)
    latest_education = get_latest_education_item(info)
    fields = [
        personal_info.get("full_name", ""),
        personal_info.get("email", ""),
        personal_info.get("phone_number", ""),
        personal_info.get("current_location", ""),
        personal_info.get("desired_position", ""),
        latest_education.get("school_name", ""),
        latest_education.get("major", ""),
        latest_experience.get("company", ""),
        latest_experience.get("position", ""),
        ", ".join(parse_skills(info)),
        ", ".join(info.get("extracted_keywords", [])),
    ]
    return "\n".join(value for value in fields if normalize_whitespace(value))


def hash_text(value: str | None) -> str:
    value = value or ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_resume_identity(info: dict, resume_content: str | list[str] | None = None) -> dict:
    personal_info = info.get("personal_info", {})
    latest_experience = get_latest_experience_item(info)
    latest_education = get_latest_education_item(info)
    canonical_cv_text = build_canonical_cv_text(info)
    resume_content_text = (
        "\n".join(resume_content)
        if isinstance(resume_content, list)
        else (resume_content or "")
    )

    return {
        "full_name": personal_info.get("full_name", ""),
        "full_name_normalized": normalize_name(personal_info.get("full_name")),
        "email": personal_info.get("email", ""),
        "email_normalized": normalize_email(personal_info.get("email")),
        "phone_number": personal_info.get("phone_number", ""),
        "phone_normalized": normalize_phone(personal_info.get("phone_number")),
        "linkedin_url": personal_info.get("linkedin_url", ""),
        "linkedin_normalized": normalize_linkedin(personal_info.get("linkedin_url")),
        "github_url": personal_info.get("github_url", ""),
        "github_normalized": normalize_github(personal_info.get("github_url")),
        "year_of_birth": get_yob(personal_info),
        "current_location": personal_info.get("current_location", ""),
        "location_normalized": normalize_location(
            personal_info.get("current_location")
        ),
        "desired_position": personal_info.get("desired_position", ""),
        "desired_positions": parse_array(personal_info.get("desired_position", "")),
        "years_of_experience": parse_years_of_experience(
            personal_info.get("year_of_experience", "")
        ),
        "latest_company": latest_experience.get("company", ""),
        "latest_company_normalized": normalize_text(latest_experience.get("company")),
        "latest_position": latest_experience.get("position", ""),
        "latest_position_normalized": normalize_text(latest_experience.get("position")),
        "latest_school": latest_education.get("school_name", ""),
        "latest_school_normalized": normalize_text(latest_education.get("school_name")),
        "latest_degree": latest_education.get("degree", ""),
        "latest_degree_normalized": normalize_text(latest_education.get("degree")),
        "skills": parse_skills(info),
        "keywords": info.get("extracted_keywords", []),
        "canonical_cv_text": canonical_cv_text,
        "content_hash": hash_text(normalize_whitespace(resume_content_text)),
        "canonical_hash": hash_text(normalize_whitespace(canonical_cv_text)),
    }


def calculate_experience_yob_score(candidate_a: dict, candidate_b: dict) -> float:
    score = 0.0
    matched = 0

    yob_a = candidate_a.get("year_of_birth")
    yob_b = candidate_b.get("year_of_birth")
    if yob_a and yob_b:
        matched += 1
        score += 1.0 if yob_a == yob_b else 0.0

    years_a = candidate_a.get("years_of_experience")
    years_b = candidate_b.get("years_of_experience")
    if years_a is not None and years_b is not None:
        matched += 1
        diff = abs(years_a - years_b)
        if diff == 0:
            score += 1.0
        elif diff <= 1:
            score += 0.8
        elif diff <= 2:
            score += 0.5

    if matched == 0:
        return 0.0
    return score / matched


def build_structured_duplicate_score(
    incoming: dict, existing: dict, vector_similarity: float = 0.0
) -> dict:
    breakdown = {
        "name": similarity_ratio(
            incoming.get("full_name_normalized"), existing.get("full_name_normalized")
        ),
        "company_position": (
            similarity_ratio(
                incoming.get("latest_company_normalized"),
                existing.get("latest_company_normalized"),
            )
            + similarity_ratio(
                incoming.get("latest_position_normalized"),
                existing.get("latest_position_normalized"),
            )
        )
        / 2,
        "school_degree": (
            similarity_ratio(
                incoming.get("latest_school_normalized"),
                existing.get("latest_school_normalized"),
            )
            + similarity_ratio(
                incoming.get("latest_degree_normalized"),
                existing.get("latest_degree_normalized"),
            )
        )
        / 2,
        "location": similarity_ratio(
            incoming.get("location_normalized"), existing.get("location_normalized")
        ),
        "skills": jaccard_similarity(
            incoming.get("skills", []), existing.get("skills", [])
        ),
        "experience_yob": calculate_experience_yob_score(incoming, existing),
        "vector": max(0.0, vector_similarity),
    }
    weights = {
        "name": 0.35,
        "company_position": 0.20,
        "school_degree": 0.10,
        "location": 0.10,
        "skills": 0.15,
        "experience_yob": 0.10,
    }
    score = sum(breakdown[key] * weights[key] for key in weights)
    hybrid_score = round((score * 0.8) + (breakdown["vector"] * 0.2), 4)
    return {
        "score": hybrid_score,
        "score_without_vector": round(score, 4),
        "vector_similarity": round(breakdown["vector"], 4),
        "breakdown": {key: round(value, 4) for key, value in breakdown.items()},
    }


def compare_resume_versions(incoming: dict, existing: dict) -> dict:
    if incoming.get("content_hash") and incoming.get("content_hash") == existing.get(
        "content_hash"
    ):
        return {
            "same_version": True,
            "version_similarity": 1.0,
            "reason": "content_hash_match",
        }

    canonical_similarity = similarity_ratio(
        incoming.get("canonical_cv_text"), existing.get("canonical_cv_text")
    )
    skill_similarity = jaccard_similarity(
        incoming.get("skills", []), existing.get("skills", [])
    )
    company_similarity = similarity_ratio(
        incoming.get("latest_company_normalized"),
        existing.get("latest_company_normalized"),
    )
    position_similarity = similarity_ratio(
        incoming.get("latest_position_normalized"),
        existing.get("latest_position_normalized"),
    )
    same_version = canonical_similarity >= 0.995 and (
        skill_similarity >= 0.95
        and company_similarity >= 0.95
        and position_similarity >= 0.95
    )
    return {
        "same_version": same_version,
        "version_similarity": round(canonical_similarity, 4),
        "reason": "canonical_similarity",
    }


def get_age(info: dict) -> int | None:
    age = info.get("age")
    if age:
        try:
            digits = ''.join(filter(str.isdigit, age))
            return int(digits)
        except (ValueError, TypeError):
            pass
    yob = info.get("year_of_birth")
    if yob:
        try:
            yob = int(yob)
            current_year = datetime.date.today().year
            return current_year - yob
        except (ValueError, TypeError):
            pass

    return None

def get_yob(info: dict) -> int | None:
    if "year_of_birth" in info:
        try:
            return int(info["year_of_birth"])
        except (ValueError, TypeError):
            pass
    age = info.get("age")
    if age:
        try:
            digits = ''.join(filter(str.isdigit, age))
            int_age = int(digits)
            current_year = datetime.date.today().year
            return current_year - int_age
        except (ValueError, TypeError):
            pass

    return None



def convert_resume_format(info):
    # ==== Personal Info ====
    p = info["personal_info"]
    personalInfo = {
        "fullName": p.get("full_name", ""),
        "phoneNumber": p.get("phone_number", ""),
        "yearOfBirth": get_yob(p),
        "nationality": p.get("nationality", ""),
        "age": get_age(p),
        "currentLocation": p.get("current_location", ""),
        "yearsOfExperience": parse_years_of_experience(p.get("year_of_experience", "")),
        "availableDate": p.get("available_date", ""),
        "desiredPositions": parse_array(p.get("desired_position", "")),
        "expectedSalary": {
            "min": parse_float(p.get("expected_salary_min", "")),
            "max": parse_float(p.get("expected_salary_max", "")),
        },
        "cvUrl": "",
        "coverLetterUrl": p.get("cover_letter_url", ""),
        "languages": p.get("languages", []),
    }

    # ==== Education ====
    education = []
    for edu in info.get("education", []):
        startDate, endDate = convert_duration_to_dates(edu.get("duration", ""))
        education.append(
            {
                "schoolName": edu.get("school_name", ""),
                "major": edu.get("major", ""),
                "degree": edu.get("degree", ""),
                "startDate": startDate,
                "endDate": endDate,
                # "description": edu.get("summary_education", "")
            }
        )

    # ==== Certificate ====
    certificates = []
    for cer in info.get("certificates", []):
        certificates.append(
            {
                "name": cer.get("certificate_name", ""),
                "issuer": cer.get("issuer", ""),
                "issedDate": cer.get("issued_date", ""),
                "fileUrl": cer.get("file_url", ""),
            }
        )

    # ==== Skills ====
    skills = []
    for s in info["skills"]:
        skills.append(
            {
                "name": s.get("skill_name", ""),
                # "level": s.get("summary_skill", ""),  # mapping if needed
                "level": None,
            }
        )

    # ==== Experiences ====
    experiences = []
    for e in info.get("experience", []):
        # if not e.get("company"):
        #     continue
        startDate, endDate = convert_duration_to_dates(e.get("duration", ""))
        company = (e.get("company") or "").strip()
        position = (e.get("position") or "").strip()
        jobDescription = (e.get("job_description") or "").strip()
        if not company and not position and not startDate and not endDate and not jobDescription:
            continue
        experiences.append(
            {
                # "company": e.get("company", ""),
                # "position": e.get("position", ""),
                # "startDate": startDate,
                # "endDate": endDate,
                "company": company,
                "position": position,
                "startDate": startDate,
                "endDate": endDate,
                "jobDescription": e.get("job_description", "")
            }
        )

    # ==== Projects =====
    project = []
    for p in info.get("project", []):
        startDate, endDate = convert_duration_to_dates(p.get("duration", ""))
        project.append(
            {
                "projectName": p.get("proj_name", ""),
                "projectCompany": p.get("proj_company", ""),
                "projectPosition": p.get("proj_position", ""),
                # "startDate": startDate,
                # "endDate": endDate,
                "projectDuration": p.get("duration", ""),
                "projectTech": p.get("proj_tech", ""),
                "projectDescription": p.get("proj_description", ""),
            }
        )
    # ==== Build convert format ====
    convert_json = {
        "personalInfo": personalInfo,
        "education": education,
        "certificates": certificates,
        "skills": skills,
        "experiences": experiences,
        "project": project,
    }

    return convert_json


def convert_jd_format():
    return


if __name__ == "__main__":
    input_folder = "/home/yaiba/Downloads/Telegram Desktop/results"
    output_folder = "output_jsons"
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            with open(input_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            new_json = convert_resume_format(raw_json)
            # print(new_json)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(new_json, f, ensure_ascii=False, indent=2)

            # print(f"Converted: {file_name} -> {output_path}")
