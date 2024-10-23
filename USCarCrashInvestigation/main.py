from utils import helpers
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *


class USVehicleCrashInvestigation:
    def __init__(self, config_file):
        zipped_files_path = helpers.read_config(config_file).get("ZippedInput")
        extracted_files_path = helpers.read_config(config_file).get("ExtractTo")
        helpers.extract_zipped_content(zipped_files_path, extracted_files_path)
        extracted_inputs = helpers.read_config(config_file).get("ExtractedInput")
        self.df_charges_use = helpers.deduplicate_df(
            helpers.load_csv(
                spark, extracted_files_path + extracted_inputs.get("Charges")
            )
        )
        self.df_damages_use = helpers.deduplicate_df(
            helpers.load_csv(
                spark, extracted_files_path + extracted_inputs.get("Damages")
            )
        )
        self.df_endorse_use = helpers.deduplicate_df(
            helpers.load_csv(
                spark, extracted_files_path + extracted_inputs.get("Endorse")
            )
        )
        self.df_primary_person_use = helpers.deduplicate_df(
            helpers.load_csv(
                spark, extracted_files_path + extracted_inputs.get("PrimaryPerson")
            )
        )
        self.df_restrict_use = helpers.deduplicate_df(
            helpers.load_csv(
                spark, extracted_files_path + extracted_inputs.get("Restrict")
            )
        )
        self.df_units_use = helpers.deduplicate_df(
            helpers.load_csv(
                spark, extracted_files_path + extracted_inputs.get("Units")
            )
        )

    def analytics_1(self, output_path, output_format):
        """
        Provides result for Analytics 1 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: count(int) of dataframe
        """
        out_df = (
            self.df_primary_person_use.filter('PRSN_GNDR_ID == "MALE" and DEATH_CNT==1')
            .select(["CRASH_ID"])
            .groupBy("CRASH_ID")
            .agg(count("*").alias("count"))
            .orderBy(desc("count"))
            .filter("count > 2")
        )
        helpers.write_output_with_format(out_df, output_path, output_format)

        return out_df.count()

    def analytics_2(self, output_path, output_format):
        """
        Provides result for Analytics 2 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: count(int) of dataframe
        """
        out_df = self.df_units_use.filter(
            self.df_units_use.VEH_BODY_STYL_ID.contains("MOTORCYCLE")
        ).dropDuplicates()
        helpers.write_output_with_format(out_df, output_path, output_format)

        return out_df.count()

    def analytics_3(self, output_path, output_format):
        """
        Provides result for Analytics 3 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: city(str) column dataframe's column
        """
        driver_death_airbag_not_deployed_df = (
            self.df_primary_person_use.filter(
                (col("DEATH_CNT") == 1)
                & (col("PRSN_AIRBAG_ID").isin("NOT DEPLOYED"))
                & col("PRSN_TYPE_ID").isin("DRIVER")
            )
            .select("CRASH_ID", "PRSN_AIRBAG_ID", "PRSN_TYPE_ID")
            .distinct()
        )
        vehicle_make_df = self.df_units_use.select("CRASH_ID", "VEH_MAKE_ID").distinct()

        out_df = (
            driver_death_airbag_not_deployed_df.join(
                vehicle_make_df, on=["CRASH_ID"], how="inner"
            )
            .groupBy("VEH_MAKE_ID")
            .count()
            .orderBy(desc("count"))
            .select(["VEH_MAKE_ID"])
            .limit(5)
        )

        helpers.write_output_with_format(out_df, output_path, output_format)

        out_df.show(truncate=False)

    def analytics_4(self, output_path, output_format):
        """
        Provides result for Analytics 4 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: comma separated string(str)
        """
        driver_details_df = (
            self.df_primary_person_use.select(["CRASH_ID", "DRVR_LIC_TYPE_ID"])
            .filter(
                (~col("DRVR_LIC_TYPE_ID").isin("NA", "UNKNOWN", "UNLICENSED"))
                & (col("PRSN_TYPE_ID").isin("DRIVER"))
            )
            .distinct()
        )
        charges_details_df = (
            self.df_charges_use.select(["CRASH_ID", "CHARGE"])
            .filter(upper(col("CHARGE")).contains("HIT AND RUN"))
            .distinct()
        )
        out_df = driver_details_df.join(
            charges_details_df, on=["CRASH_ID"], how="inner"
        )

        helpers.write_output_with_format(out_df, output_path, output_format)

        return out_df.count()

    def analytics_5(self, output_path, output_format):
        """
        Provides result for Analytics 5 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: none
        """
        out_df = (
            self.df_primary_person_use.select("DRVR_LIC_STATE_ID", "PRSN_GNDR_ID")
            .filter(~col("PRSN_GNDR_ID").isin("FEMALE"))
            .groupBy("DRVR_LIC_STATE_ID")
            .count()
            .orderBy(desc("count"))
            .select("DRVR_LIC_STATE_ID")
            .limit(1)
            
        )

        helpers.write_output_with_format(out_df, output_path, output_format)
        return out_df.show()

    def analytics_6(self, output_path, output_format):
        """
        Provides result for Analytics 6 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: none
        """
        out_df = (
            self.df_units_use.filter(self.df_units_use.VEH_MAKE_ID != "NA")
            .withColumn(
                "TOT_CASUALTIES_CNT", self.df_units_use[35] + self.df_units_use[36]
            )
            .groupby("VEH_MAKE_ID")
            .sum("TOT_CASUALTIES_CNT")
            .withColumnRenamed("sum(TOT_CASUALTIES_CNT)", "TOT_CASUALTIES")
            .orderBy(col("TOT_CASUALTIES").desc())
        )
        out_df = out_df.limit(5).subtract(out_df.limit(2))

        helpers.write_output_with_format(out_df, output_path, output_format)
        return out_df.select("VEH_MAKE_ID").show()

    def analytics_7(self, output_path, output_format):
        """
        Provides result for Analytics 5 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: none
        """
        joined_df = self.df_primary_person_use.join(
            self.df_units_use,
            self.df_primary_person_use.CRASH_ID == self.df_units_use.CRASH_ID,
            "inner",
        )
        out_df = (
            joined_df.filter(
                (
                    ~joined_df.VEH_BODY_STYL_ID.isin(
                        [
                            "NA",
                            "UNKNOWN",
                            "NOT REPORTED",
                            "OTHER  (EXPLAIN IN NARRATIVE)",
                        ]
                    )
                )
                & (~joined_df.PRSN_ETHNICITY_ID.isin(["NA", "UNKNOWN"]))
            )
            .groupBy("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID")
            .count()
            .withColumn(
                "row_number",
                row_number().over(
                    Window.partitionBy("VEH_BODY_STYL_ID").orderBy(
                        col("VEH_BODY_STYL_ID").desc(), col("count").desc()
                    )
                ),
            )
            .filter(col("row_number") == 1)
            .orderBy(col("count").desc())
            .drop("row_number", "count")
        )

        helpers.write_output_with_format(out_df, output_path, output_format)
        return out_df.show(truncate=False)

    def analytics_8(self, output_path, output_format):
        """
        Provides result for Analytics 6 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: none
        """
        out_df = (
            self.df_units_use.join(
                self.df_primary_person_use, on=["CRASH_ID"], how="inner"
            )
            .dropna(subset=["DRVR_ZIP"])
            .filter(
                col("CONTRIB_FACTR_1_ID").contains("ALCOHOL")
                | col("CONTRIB_FACTR_2_ID").contains("ALCOHOL")
            )
            .groupby("DRVR_ZIP")
            .count()
            .orderBy(col("count").desc())
            .limit(5)
        )

        helpers.write_output_with_format(out_df, output_path, output_format)
        return ",".join([row[0] for row in out_df.collect()])

    def analytics_9(self, output_path, output_format):
        """
        Provides result for Analytics 7 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: count(int) of dataframe
        """
        out_df = (
            self.df_units_use.join(self.df_damages_use, on=["CRASH_ID"], how="inner")
            .filter(
                (col("DAMAGED_PROPERTY").rlike("NO DAMAGE"))
                | (col("DAMAGED_PROPERTY").rlike("NONE"))
            )
            .filter(
                (col("VEH_DMAG_SCL_1_ID").rlike("[5-9]"))
                | (col("VEH_DMAG_SCL_2_ID").rlike("[5-9]"))
            )
            .filter((col("FIN_RESP_TYPE_ID") == "PROOF OF LIABILITY INSURANCE"))
        )

        helpers.write_output_with_format(out_df, output_path, output_format)
        return out_df.select("CRASH_ID").distinct().count()

    def analytics_10(self, output_path, output_format):
        """
        Provides result for Analytics 8 - See main method for more details
        :param output_path: output path of the output to be stored
        :param output_format: format of the output to be stored as
        :return: comma separated string(str)
        """
        top_25_states = [
            row[0]
            for row in self.df_units_use.filter(
                col("VEH_LIC_STATE_ID").cast("int").isNull()
            )
            .groupby("VEH_LIC_STATE_ID")
            .count()
            .orderBy(col("count").desc())
            .limit(25)
            .collect()
        ]
        top_10_used_vcolors = [
            row[0]
            for row in self.df_units_use.filter(self.df_units_use.VEH_COLOR_ID != "NA")
            .groupby("VEH_COLOR_ID")
            .count()
            .orderBy(col("count").desc())
            .limit(10)
            .collect()
        ]
        out_df = (
            self.df_charges_use.join(
                self.df_primary_person_use, on=["CRASH_ID"], how="inner"
            )
            .join(self.df_units_use, on=["CRASH_ID"], how="inner")
            .filter(self.df_charges_use.CHARGE.contains("SPEED"))
            .filter(
                self.df_primary_person_use.DRVR_LIC_TYPE_ID.isin(
                    ["DRIVER LICENSE", "COMMERCIAL DRIVER LIC."]
                )
            )
            .filter(self.df_units_use.VEH_COLOR_ID.isin(top_10_used_vcolors))
            .filter(self.df_units_use.VEH_LIC_STATE_ID.isin(top_25_states))
            .groupby("VEH_MAKE_ID")
            .count()
            .orderBy(col("count").desc())
            .limit(5)
        )
        out_list = helpers.rdd_ops_list(out_df, "VEH_MAKE_ID")
        helpers.write_output_with_format(out_df, output_path, output_format)
        return out_list


if __name__ == "__main__":

    spark = SparkSession.builder.appName("USVehicleCrashInvestigation").getOrCreate()

    config_file = "config.json"
    spark.sparkContext.setLogLevel("ERROR")

    accidents = USVehicleCrashInvestigation(config_file)
    # Specify the output location via config.json
    output_files = helpers.read_config(config_file).get("Output")
    # Specify the output format for the files
    output_format = helpers.read_config(config_file).get("OutputFormat")

    # Analytics 1: Find the number of crashes (accidents) in which number of males killed are greater than 2?
    print(
        "Analytics 1 - Number of crashes (accidents) in which number of males killed are greater than 2 are: ",
        accidents.analytics_1(output_files["Analytics 1"], output_format),
    )

    # Analysis 2: How many two wheelers are booked for crashes?
    print(
        "Analytics 2 - Number of two-wheelers that are booked for crashes are: ",
        accidents.analytics_2(output_files["Analytics 2"], output_format),
    )

    # Analysis 3: Determine the Top 5 Vehicle Makes of the cars present in the crashes in which driver died and Airbags did not deploy
    print(
        "Analytics 3 - Top 5 Vehicle Makes of the cars present in the crashes in which driver died and Airbags did not deploy: ",
        accidents.analytics_3(output_files["Analytics 3"], output_format),
    )

    # Analysis 4: Determine number of Vehicles with driver having valid licences involved in hit and run?
    print(
        "Analytics 4 - Number of Vehicles with driver having valid licences involved in hit and run are: ",
        accidents.analytics_4(output_files["Analytics 4"], output_format),
    )

    # Analysis 5: Which state has highest number of accidents in which females are not involved
    print(
        "Analytics 5 - State that has highest number of accidents in which females are not involved is: "
    )
    accidents.analytics_5(output_files["Analytics 5"], output_format)

    # Analysis 6: Which are the Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death
    print(
        "Analytics 6 - Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death are: ",
        accidents.analytics_6(output_files["Analytics 6"], output_format),
    )

    # For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
    print("Analytics 7 - Top ethnic user group for each vehicle body style: ")
    accidents.analytics_7(output_files["Analytics 5"], output_format)

    # Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)
    print(
        "Analytics 8 - Top 5 zip codes with highest crashes involving Alcohol as contributing factor are: ",
        accidents.analytics_8(output_files["Analytics 8"], output_format),
    )

    # Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
    print(
        "Analytics 9 - Count of distinct Crash IDs where no property damage was observed and car has insurance is: ",
        accidents.analytics_9(output_files["Analytics 9"], output_format),
    )

    # Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences,
    # has licensed Drivers, uses top 10 used vehicle colours and
    # has car licensed with the Top 25 states with highest number of offences (to be deduced from the data)
    print(
        """Analytics 10 - Top 5 vehicle makes where drivers are charged with speeding, driver has license,
     uses to 10 vehicle colors and is licensed with the to 25 states with highest nuber of offences are: """,
        accidents.analytics_10(output_files["Analytics 10"], output_format),
    )

    spark.stop()

