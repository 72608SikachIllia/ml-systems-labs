
import print_versions
import ingest
import validate
import preprocess
import features
import label_audit
import finalize
import make_report

if __name__ == "__main__":
    for step in (print_versions, ingest, validate, preprocess, features, label_audit, finalize,
                 make_report):
        print(f"\n=== {step.__name__} ===")
        step.main()
