CREATE TABLE IF NOT EXISTS finance_bookings (
    booking_date DATE NOT NULL,
    gross_bookings DECIMAL(14, 2) NOT NULL,
    region TEXT NOT NULL
);

INSERT INTO finance_bookings (booking_date, gross_bookings, region) VALUES
    ('2026-07-05', 125000.00, 'North'),
    ('2026-08-15', 148500.00, 'South'),
    ('2026-09-21', 139750.00, 'North');
