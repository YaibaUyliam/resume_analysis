import json
import re
import os
import datetime


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
        .strip()
    )
    if "-" not in duration:
        return duration, None
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
        return float(value)
    except ValueError:
        return None


def get_age(info: dict) -> int | None:
    if "age" in info:
        try:
            return int(info["age"])
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


def convert_resume_format(info):
    # ==== Personal Info ====
    p = info["personal_info"][0]
    personalInfo = {
        "fullName": p.get("full_name", ""),
        "phoneNumber": p.get("phone_number", ""),
        "yearOfBirth": p.get("year_of_birth", ""),
        "nationality": p.get("nationality", ""),
        "age": get_age(p),
        "currentLocation": p.get("current_location", ""),
        "yearsOfExperience": parse_years_of_experience(p.get("year_of_experience", "")),
        "availableDate": p.get("available_date", ""),
        "desiredPositions": p.get("desired_positions", []),
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
        if not company and not position and not startDate and not endDate:
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
                # "description": e.get("job_description", "")
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
        "skills": skills,
        "experiences": experiences,
        "project": project,
    }

    return convert_json


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
