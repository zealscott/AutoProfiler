
def parsing_function_response(execute_results):
    """
    Parsing the response from the model
    """
    # Splitting by '[STATUS]'
    parts_after_status = execute_results.split("[STATUS]: ")
    
    # Initializing variables to hold the parts before '[STATUS]', between '[STATUS]' and '[RESULT]', and after '[RESULT]'
    part_between_status_and_result = ""
    part_after_result = ""
    
    # Check if there's a part after '[STATUS]'
    if len(parts_after_status) > 1:
        # Splitting the part after '[STATUS]' by '[RESULT]'
        parts_after_result = parts_after_status[1].split("[RESULT]: ")
        
        # Extracting the part between '[STATUS]' and '[RESULT]'
        part_between_status_and_result = parts_after_result[0].strip()
        
        # Check if there's a part after '[RESULT]'
        if len(parts_after_result) > 1:
            # Extracting the part after '[RESULT]'
            part_after_result = parts_after_result[1].strip()
    
    if "SUCCESS" in part_between_status_and_result:
        part_between_status_and_result = "success"
    else:
        part_between_status_and_result = "fail"
    
    return part_between_status_and_result, part_after_result