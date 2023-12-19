import numpy as np

def bobot(bidang):
    if bidang == "Web Developer":
        return {
            "pengalaman_kerja": 0.3,
            "skill_sertifikat": 0.25,
            "pengalaman_organisasi": 0.2,
            "ipk": 0.1,
            "umur": 0.05,
            "lokasi_kerja": 0.1
        }
    elif bidang == "Marketing":
        return {
            "pengalaman_kerja": 0.2,
            "skill_sertifikat": 0.2,
            "pengalaman_organisasi": 0.2,
            "ipk": 0.1,
            "umur": 0.1,
            "lokasi_kerja": 0.2
        }
    elif bidang == "UI/UX":
        return {
            "pengalaman_kerja": 0.15,
            "skill_sertifikat": 0.25,
            "pengalaman_organisasi": 0.1,
            "ipk": 0.1,
            "umur": 0.1,
            "lokasi_kerja": 0.3
        }
    elif bidang == "Data Analyst":
        return {
            "pengalaman_kerja": 0.25,
            "skill_sertifikat": 0.25,
            "pengalaman_organisasi": 0.1,
            "ipk": 0.1,
            "umur": 0.1,
            "lokasi_kerja": 0.2
        }
    elif bidang == "Mobile Developer":
        return {
            "pengalaman_kerja": 0.2,
            "skill_sertifikat": 0.25,
            "pengalaman_organisasi": 0.1,
            "ipk": 0.1,
            "umur": 0.1,
            "lokasi_kerja": 0.25
        }

def averageValue(criterias):
    avg = {
        "pengalaman_kerja": 0,
        "skill_sertifikat": 0,
        "pengalaman_organisasi": 0,
        "ipk": 0,
        "umur": 0,
        "lokasi_kerja": 0
    }
    for c in criterias:
        avg["pengalaman_kerja"] += float(c["pengalaman_kerja"])
        avg["skill_sertifikat"] += float(c["skill_sertifikat"])
        avg["pengalaman_organisasi"] += float(c["pengalaman_organisasi"])
        avg["ipk"] += float(c["ipk"])
        avg["umur"] += float(c["umur"])
        avg["lokasi_kerja"] += float(c["lokasi_kerja"])

    avg["pengalaman_kerja"] /= len(criterias)
    avg["skill_sertifikat"] /= len(criterias)
    avg["pengalaman_organisasi"] /= len(criterias)
    avg["ipk"] /= len(criterias)
    avg["umur"] /= len(criterias)
    avg["lokasi_kerja"] /= len(criterias)

    return avg

def pda(criterias, avg):
    temp_criterias = []
    for i, c in enumerate(criterias):
        temp = {
            "pengalaman_kerja": 0,
            "skill_sertifikat": 0,
            "pengalaman_organisasi": 0,
            "ipk": 0,
            "umur": 0,
            "lokasi_kerja": 0,
            "name": c["name"]
        }
        temp_criterias.append(temp)

        temp_criterias[i]["pengalaman_kerja"] = max(0, (float(c["pengalaman_kerja"]) - avg["pengalaman_kerja"]) / avg["pengalaman_kerja"])
        temp_criterias[i]["skill_sertifikat"] = max(0, (float(c["skill_sertifikat"]) - avg["skill_sertifikat"]) / avg["skill_sertifikat"])
        temp_criterias[i]["pengalaman_organisasi"] = max(0, (float(c["pengalaman_organisasi"]) - avg["pengalaman_organisasi"]) / avg["pengalaman_organisasi"])
        temp_criterias[i]["ipk"] = max(0, (float(c["ipk"]) - avg["ipk"]) / avg["ipk"])
        temp_criterias[i]["umur"] = max(0, (avg["umur"] - float(c["umur"])) / avg["umur"])
        temp_criterias[i]["lokasi_kerja"] = max(0, (avg["lokasi_kerja"] - float(c["lokasi_kerja"])) / avg["lokasi_kerja"])
    
    return temp_criterias

def nda(criterias, avg):
    temp_criterias = []
    for i, c in enumerate(criterias):
        temp = {
            "pengalaman_kerja": 0,
            "skill_sertifikat": 0,
            "pengalaman_organisasi": 0,
            "ipk": 0,
            "umur": 0,
            "lokasi_kerja": 0,
            "name": c["name"]
        }
        temp_criterias.append(temp)

        temp_criterias[i]["pengalaman_kerja"] = max(0, (avg["pengalaman_kerja"] - float(c["pengalaman_kerja"])) / avg["pengalaman_kerja"])
        temp_criterias[i]["skill_sertifikat"] = max(0, (avg["skill_sertifikat"] - float(c["skill_sertifikat"])) / avg["skill_sertifikat"])
        temp_criterias[i]["pengalaman_organisasi"] = max(0, (avg["pengalaman_organisasi"] - float(c["pengalaman_organisasi"])) / avg["pengalaman_organisasi"])
        temp_criterias[i]["ipk"] = max(0, (avg["ipk"] - float(c["ipk"])) / avg["ipk"])
        temp_criterias[i]["umur"] = max(0, (float(c["umur"]) - avg["umur"]) / avg["umur"])
        temp_criterias[i]["lokasi_kerja"] = max(0, (float(c["lokasi_kerja"]) - avg["lokasi_kerja"]) / avg["lokasi_kerja"])
    
    return temp_criterias

def sp(pda, bidang):
    w = bobot(bidang)
    result = []
    for p in pda:
        total = 0
        total += p["pengalaman_kerja"] * w["pengalaman_kerja"]
        total += p["skill_sertifikat"] * w["skill_sertifikat"]
        total += p["pengalaman_organisasi"] * w["pengalaman_organisasi"]
        total += p["ipk"] * w["ipk"]
        total += p["umur"] * w["umur"]
        total += p["lokasi_kerja"] * w["lokasi_kerja"]
        result.append(total)
    
    return result

def sn(nda, bidang):
    w = bobot(bidang)
    result2 = []
    for n in nda:
        total2 = 0
        total2 += n["pengalaman_kerja"] * w["pengalaman_kerja"]
        total2 += n["skill_sertifikat"] * w["skill_sertifikat"]
        total2 += n["pengalaman_organisasi"] * w["pengalaman_organisasi"]
        total2 += n["ipk"] * w["ipk"]
        total2 += n["umur"] * w["umur"]
        total2 += n["lokasi_kerja"] * w["lokasi_kerja"]
        result2.append(total2)
    
    return result2

def nsp(sp):
    max_sp = max(sp)
    result = []
    for s in sp:
        result.append(s / max_sp)
    
    return result

def nsn(sn):
    max_sn = max(sn)
    result = []
    for s in sn:
        result.append(1 - (s / max_sn))
    
    return result

def ranking(nsp, nsn, alternative):
    # rank = {}
    rank = []
    for i, np in enumerate(nsp):
        data = 0.5 * (nsp[i] + nsn[i])
        rank.append({
            "name": alternative[i],
            "value": data
        })
        # rank[alternative[i]] = data

    # sorted_data = dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))

    return rank