import pandas as pd
import numpy as np

class AdvancedStudentAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        median_score = self.df['project_score'].median()
        self.df['project_score'] = self.df['project_score'].fillna(median_score)
        subjects = ['math', 'physics', 'cs', 'project_score']
        self.df['average_grade'] = self.df[subjects].mean(axis=1)

        conditions = [self.df['average_grade'] >= 85, (self.df['average_grade'] >= 70) & (self.df['average_grade'] < 85), self.df['average_grade'] < 70]
        choices = ['high', 'medium', 'low']
        self.df['performance_level'] = np.select(conditions, choices, default='medium')

        risk_conditions = [(self.df['attendance'] < 60) | (self.df['average_grade'] < 65), (self.df['attendance'] >= 60) & (self.df['attendance'] <= 75), True]
        risk_choices = ['high risk', 'medium risk', 'low risk']
        self.df['risk_level'] = np.select(risk_conditions, risk_choices, default='low risk')

    def top_students(self, n: int) -> pd.DataFrame:
        top = self.df.nlargest(n, 'average_grade')[['name', 'group', 'average_grade']]
        return top

    def group_stats(self) -> pd.DataFrame:
        stats = self.df.groupby('group').agg(avg_grade=('average_grade', 'mean'), avg_attendance=('attendance', 'mean'), student_count=('name', 'count')).reset_index()
        return stats

    def at_risk_students(self) -> pd.DataFrame:
        return self.df[self.df['risk_level'] == 'high risk']

    def scholarship_analysis(self) -> pd.DataFrame:
        analysis = self.df.groupby('scholarship')[['average_grade', 'attendance']].mean()
        return analysis

    def city_performance(self):
        city_avg = self.df.groupby('city')['average_grade'].mean()
        best_city = city_avg.idxmax()
        worst_city = city_avg.idxmin()
        return best_city, worst_city

    def hidden_top_students(self) -> pd.DataFrame:
        return self.df[(self.df['average_grade'] > 85) & (self.df['scholarship'] == False)]

    def lazy_geniuses(self) -> pd.DataFrame:
        return self.df[(self.df['average_grade'] > 85) & (self.df['attendance'] < 60)]

    def full_analysis(self) -> dict:

        top3 = self.top_students(3).to_dict('records')
        group_stats_df = self.group_stats()
        group_stats = group_stats_df.to_dict('records')

        high_risk_count = len(self.at_risk_students())
        hidden_top_count = len(self.hidden_top_students())
        lazy_geniuses_count = len(self.lazy_geniuses())

        best_city, worst_city = self.city_performance()
        scholarship_df = self.scholarship_analysis()
        scholarship_dict = scholarship_df.to_dict()

        result = {
            'top_3_students': top3,
            'group_statistics': group_stats,
            'high_risk_students_count': high_risk_count,
            'hidden_top_students_count': hidden_top_count,
            'lazy_geniuses_count': lazy_geniuses_count,
            'best_city': best_city,
            'worst_city': worst_city,
            'scholarship_comparison': scholarship_dict
        }
        return result


if __name__ == "__main__":
    df = pd.read_csv('students_extended.csv')
    analytics = AdvancedStudentAnalytics(df)
    print("Топ 3 Студента")
    print(analytics.top_students(3).to_string(index=False))
    print()
    print(analytics.group_stats().to_string(index=False))
    print()
    print('С риском(первые 5)')
    risk_df = analytics.at_risk_students()[['name', 'group', 'average_grade', 'attendance', 'risk_level']]
    if len(risk_df) > 0:
        print(risk_df.head().to_string(index=False))
        print(f"Всего студентов с высоким риском: {len(risk_df)}")
    else:
        print("Нет студентов с высоким риском.")
    print()
    print("Степендианты(сравнение)")
    scholarship_comp = analytics.scholarship_analysis()
    print("Средний балл и посещаемость:")
    print(scholarship_comp.round(2).to_string())
    print()

    best, worst = analytics.city_performance()
    print("ГОРОДА")
    print(f"Лучший город по среднему баллу: {best}")
    print(f"Худший город по среднему баллу: {worst}")
    print()
    print("СКРЫТЫЕ ОТЛИЧНИКИ")
    hidden = analytics.hidden_top_students()
    if len(hidden) > 0:
        print(hidden[['name', 'group', 'average_grade']].head().to_string(index=False))
        print(f"Всего: {len(hidden)}")
    else:
        print("Нет таких студентов")
    print()
    print("ЛЕНИВЫЕ ГЕНИИ")
    lazy = analytics.lazy_geniuses()
    if len(lazy) > 0:
        print(lazy[['name', 'group', 'average_grade', 'attendance']].head().to_string(index=False))
        print(f"Всего: {len(lazy)}")
    else:
        print("Нет таких студентов")
    print()


    full = analytics.full_analysis()

    print("Топ-3 студента:")
    for s in full['top_3_students']:
        print(f"  {s['name']} (группа {s['group']}) - средний балл {s['average_grade']:.2f}")

    print("\nСтатистика по группам:")
    for g in full['group_statistics']:
        print(f"  Группа {g['group']}: средний балл = {g['avg_grade']:.2f}, "
              f"посещаемость = {g['avg_attendance']:.1f}%, студентов = {g['student_count']}")

    print(f"\nСтудентов с высоким риском: {full['high_risk_students_count']}")
    print(f"Скрытых отличников: {full['hidden_top_students_count']}")
    print(f"Ленивых гениев: {full['lazy_geniuses_count']}")
    print(f"Лучший город: {full['best_city']}")
    print(f"Худший город: {full['worst_city']}")

    print("\nСравнение стипендий:")
    sch = full['scholarship_comparison']
    avg_grade_true = sch['average_grade'].get(True, sch['average_grade'].get('True', 0))
    avg_grade_false = sch['average_grade'].get(False, sch['average_grade'].get('False', 0))
    att_true = sch['attendance'].get(True, sch['attendance'].get('True', 0))
    att_false = sch['attendance'].get(False, sch['attendance'].get('False', 0))
    print(f"  Средний балл: со стипендией = {avg_grade_true:.2f}, без = {avg_grade_false:.2f}")
    print(f"  Посещаемость:  со стипендией = {att_true:.1f}%, без = {att_false:.1f}%")