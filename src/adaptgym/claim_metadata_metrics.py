"""Reporting-only scope/strength errors with explicit claim-alignment coverage.

Typed claims align on content, never on the scope/strength labels being tested.
Open claims align only on schema-derived kind and unambiguous user citations;
their free-form names and values are not compared to reference strings.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction

from adaptgym.feature_scoring import features
from adaptgym.open_count_scoring import partition, score_open_counts, summarize_open_counts

POLICY = 'claim-metadata-content-or-evidence-v1'
KINDS = ('in_schema', 'new_feature', 'new_value')
FIELDS = ('scope', 'strength')


def _claims(profile):
    value = profile.get('claims') if isinstance(profile, dict) else None
    return value if isinstance(value, list) else []


def _kind(claim, vertical):
    if not isinstance(claim, dict) or not isinstance(claim.get('attribute'), str) or not claim['attribute'].strip():
        return None
    if 'value' not in claim or not isinstance(claim.get('operator'), str):
        return None
    typed, counts = partition({'claims': [claim]}, vertical)
    if counts['new_feature']:
        return 'new_feature'
    if counts['new_value']:
        return 'new_value'
    return 'in_schema' if typed['claims'] else None


def _user_evidence(claim, user_ids):
    evidence = claim.get('evidence', [])
    if not isinstance(evidence, list):
        return set()
    return {mid for item in evidence if isinstance(item, dict) and isinstance(item.get('message_ids'), list)
            for mid in item['message_ids'] if isinstance(mid, str) and mid in user_ids}


def _label(claim, field):
    value = claim.get(field)
    return value if isinstance(value, str) else '<missing>'


def score_claim_metadata(prediction, reference, vertical, user_message_ids=()):
    """Conditional error rates plus missing/ambiguous coverage, per claim.

    No matching decision uses scope or strength. Missing/ambiguous claims are
    unassessed, not silently correct; all-reference failure rates include them.
    A metadata error on an open claim is diagnostic only, not an open-content
    penalty or a new checkpoint-selection objective.
    """
    refs, preds = _claims(reference), _claims(prediction)
    user_ids = set(user_message_ids)
    ref_kinds = [_kind(c, vertical) for c in refs]
    if any(kind is None for kind in ref_kinds):
        raise ValueError('Invalid reference claim for metadata assessment')
    pred_kinds = [_kind(c, vertical) for c in preds]
    groups_r, groups_p = defaultdict(list), defaultdict(list)
    for target, claims, kinds in ((groups_r, refs, ref_kinds), (groups_p, preds, pred_kinds)):
        for index, (claim, kind) in enumerate(zip(claims, kinds)):
            if kind is None:
                continue
            content = frozenset(features({'claims': [claim]})) if kind == 'in_schema' else None
            target[(kind, content)].append(index)
    pairs, candidates = [], defaultdict(set)
    for key, ris in groups_r.items():
        pis = groups_p.get(key, [])
        if key[0] == 'in_schema' and len(ris) == len(pis) == 1:
            pairs.append((ris[0], pis[0], 'typed_content'))
            continue
        edges = {}
        for ri in ris:
            a = _user_evidence(refs[ri], user_ids)
            for pi in pis:
                b = _user_evidence(preds[pi], user_ids)
                overlap = a & b
                if overlap:
                    edges[ri, pi] = (a == b, Fraction(len(overlap), len(a | b)), len(overlap))
                    candidates[ri].add(pi)
        best_r, best_p = {}, {}
        for indices, by_ref, best in ((ris, True, best_r), (pis, False, best_p)):
            for index in indices:
                options = [(pair[1] if by_ref else pair[0], rank) for pair, rank in edges.items()
                           if pair[0 if by_ref else 1] == index]
                if options:
                    top = max(rank for _, rank in options)
                    best[index] = {other for other, rank in options if rank == top}
        for ri, choices in best_r.items():
            if len(choices) == 1:
                pi = next(iter(choices))
                if best_p.get(pi) == {ri}:
                    pairs.append((ri, pi, 'user_evidence'))
    assert len({r for r, _, _ in pairs}) == len(pairs) == len({p for _, p, _ in pairs})
    aligned = {ri for ri, _, _ in pairs}
    matches = [{'reference_index': ri, 'prediction_index': pi, 'kind': ref_kinds[ri], 'basis': basis,
                **{field: {'expected': _label(refs[ri], field), 'predicted': _label(preds[pi], field),
                           'error': _label(refs[ri], field) != _label(preds[pi], field)} for field in FIELDS}}
               for ri, pi, basis in sorted(pairs)]
    report = {'policy': POLICY, 'reporting_only': True,
              'expected_claims': len(refs), 'predicted_claims': len(preds),
              'matched_claims': len(matches), 'unassessed_reference_claims': len(refs) - len(matches),
              'ambiguous_reference_claims': sum(ri not in aligned and bool(candidates[ri]) for ri in range(len(refs))),
              'unmatched_prediction_claims': len(preds) - len(matches),
              'invalid_prediction_claims': sum(kind is None for kind in pred_kinds),
              'unusable_output': not (isinstance(prediction, dict) and isinstance(prediction.get('claims'), list)),
              'expected_by_kind': dict(Counter(ref_kinds)), 'matches': matches}
    for field in FIELDS:
        report[field + '_errors'] = sum(m[field]['error'] for m in matches)
        report[field + '_confusion'] = _confusion(matches, field)
    report['by_kind'] = {kind: _group_summary([report], kind) for kind in KINDS}
    report.update(_group_summary([report]))
    return report


def _confusion(matches, field):
    matrix = defaultdict(Counter)
    for match in matches:
        matrix[match[field]['expected']][match[field]['predicted']] += 1
    return {key: dict(value) for key, value in sorted(matrix.items())}


def _group_summary(reports, kind=None):
    matches = [m for r in reports for m in r['matches'] if kind is None or m['kind'] == kind]
    expected = sum(r['expected_claims'] if kind is None else r['expected_by_kind'].get(kind, 0) for r in reports)
    assessed = len(matches)
    result = {'expected_claims': expected, 'matched_claims': assessed,
              'unassessed_reference_claims': expected - assessed,
              'alignment_coverage': assessed / expected if expected else None}
    for field in FIELDS:
        errors = sum(m[field]['error'] for m in matches)
        result[field + '_errors'] = errors
        result[field + '_error_rate'] = errors / assessed if assessed else None
        result[field + '_or_unassessed_rate'] = (errors + expected - assessed) / expected if expected else None
        result[field + '_confusion'] = _confusion(matches, field)
    return result


def summarize_claim_metadata(reports):
    reports = list(reports)
    return {'policy': POLICY, 'reporting_only': True, 'turns': len(reports),
            **_group_summary(reports),
            'predicted_claims': sum(r['predicted_claims'] for r in reports),
            'ambiguous_reference_claims': sum(r['ambiguous_reference_claims'] for r in reports),
            'unmatched_prediction_claims': sum(r['unmatched_prediction_claims'] for r in reports),
            'invalid_prediction_claims': sum(r['invalid_prediction_claims'] for r in reports),
            'unusable_output_turns': sum(r['unusable_output'] for r in reports),
            'by_kind': {kind: _group_summary(reports, kind) for kind in KINDS}}


def score_error_diagnostics(prediction, reference, vertical, user_message_ids=()):
    return {'metadata': score_claim_metadata(prediction, reference, vertical, user_message_ids),
            'open_counts': score_open_counts(prediction, reference, vertical)}


def summarize_error_diagnostics(reports):
    reports = list(reports)
    unavailable = sum(r is None for r in reports)
    reports = [r for r in reports if r is not None]
    metadata = summarize_claim_metadata(r['metadata'] for r in reports)
    counts = summarize_open_counts(r['open_counts'] for r in reports)
    n = len(reports)
    return {'policy': POLICY, 'reporting_only': True, 'turns': n,
            'total_turns': n + unavailable, 'unavailable_turns': unavailable,
            'scope_err': metadata['scope_error_rate'], 'strength_err': metadata['strength_error_rate'],
            'match_cov': metadata['alignment_coverage'],
            'new_feat_err': sum(r['open_counts']['penalty_by_kind']['new_feature'] > 0 for r in reports) / n if n else None,
            'new_val_err': sum(r['open_counts']['penalty_by_kind']['new_value'] > 0 for r in reports) / n if n else None,
            'new_feat_mae': counts['penalty_by_kind']['new_feature'] / n if n else None,
            'new_val_mae': counts['penalty_by_kind']['new_value'] / n if n else None,
            'metadata': metadata, 'open_counts': counts}
