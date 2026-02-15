/**
 * Signal Catcher Web App - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(function(message) {
        setTimeout(function() {
            message.style.opacity = '0';
            message.style.transition = 'opacity 0.5s ease';
            setTimeout(function() {
                message.style.display = 'none';
            }, 500);
        }, 5000);
    });

    // Add loading state to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="btn-icon">⏳</span> Processing...';
                submitBtn.style.opacity = '0.7';
            }
        });
    });

    // Format numbers in tables
    const numericCells = document.querySelectorAll('.numeric');
    numericCells.forEach(function(cell) {
        const value = parseFloat(cell.textContent);
        if (!isNaN(value)) {
            // Add color coding for positive/negative values
            if (value > 0 && cell.textContent.includes('%')) {
                cell.classList.add('positive');
            } else if (value < 0 && cell.textContent.includes('%')) {
                cell.classList.add('negative');
            }
        }
    });

    // Highlight rows with high conviction
    const highConvictionRows = document.querySelectorAll('tr.high');
    highConvictionRows.forEach(function(row) {
        row.style.backgroundColor = 'rgba(16, 185, 129, 0.05)';
    });

    // Add tooltip functionality for signal cards
    const signalCards = document.querySelectorAll('.signal-card');
    signalCards.forEach(function(card) {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Sortable tables (basic implementation)
    const tables = document.querySelectorAll('.data-table');
    tables.forEach(function(table) {
        const headers = table.querySelectorAll('th');
        headers.forEach(function(header, index) {
            if (!header.classList.contains('no-sort')) {
                header.style.cursor = 'pointer';
                header.addEventListener('click', function() {
                    sortTable(table, index);
                });
                
                // Add sort indicator
                header.style.position = 'relative';
                header.style.paddingRight = '20px';
            }
        });
    });

    console.log('Signal Catcher Web App loaded successfully');
});

/**
 * Basic table sorting function
 * @param {HTMLTableElement} table - The table to sort
 * @param {number} column - Column index to sort by
 */
function sortTable(table, column) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    // Determine sort direction
    const header = table.querySelectorAll('th')[column];
    const isAscending = !header.classList.contains('sort-asc');
    
    // Remove sort classes from all headers
    table.querySelectorAll('th').forEach(function(th) {
        th.classList.remove('sort-asc', 'sort-desc');
    });
    
    // Add sort class to current header
    header.classList.add(isAscending ? 'sort-asc' : 'sort-desc');
    
    // Sort rows
    rows.sort(function(a, b) {
        const aVal = a.cells[column].textContent.trim();
        const bVal = b.cells[column].textContent.trim();
        
        // Try to parse as numbers
        const aNum = parseFloat(aVal.replace(/[^\d.-]/g, ''));
        const bNum = parseFloat(bVal.replace(/[^\d.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return isAscending ? aNum - bNum : bNum - aNum;
        }
        
        // Fallback to string comparison
        return isAscending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    
    // Re-append rows in new order
    rows.forEach(function(row) {
        tbody.appendChild(row);
    });
}

/**
 * Export table data to CSV
 * @param {string} tableId - ID of the table to export
 * @param {string} filename - Name for the downloaded file
 */
function exportTableToCSV(tableId, filename) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    let csv = [];
    const rows = table.querySelectorAll('tr');
    
    rows.forEach(function(row) {
        let cols = row.querySelectorAll('td, th');
        let rowData = [];
        
        cols.forEach(function(col) {
            rowData.push(col.textContent.trim());
        });
        
        csv.push(rowData.join(','));
    });
    
    downloadCSV(csv.join('\n'), filename);
}

/**
 * Download CSV content
 * @param {string} csv - CSV content
 * @param {string} filename - Name for the file
 */
function downloadCSV(csv, filename) {
    const csvFile = new Blob([csv], { type: 'text/csv' });
    const downloadLink = document.createElement('a');
    
    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = 'none';
    
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}

/**
 * Refresh data with loading indicator
 * @param {string} url - URL to refresh
 */
function refreshData(url) {
    const btn = document.querySelector('.refresh-btn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ Refreshing...';
    }
    
    window.location.href = url;
}

/**
 * Filter table rows based on input
 * @param {string} tableId - Table ID
 * @param {string} filterId - Filter input ID
 */
function filterTable(tableId, filterId) {
    const table = document.getElementById(tableId);
    const filter = document.getElementById(filterId);
    
    if (!table || !filter) return;
    
    const rows = table.querySelectorAll('tbody tr');
    const term = filter.value.toLowerCase();
    
    rows.forEach(function(row) {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(term) ? '' : 'none';
    });
}

// Console greeting
console.log('%c Signal Catcher ', 'background: #2563eb; color: white; font-size: 24px; padding: 10px; border-radius: 5px;');
console.log('%c CSEMA Trading System ', 'background: #10b981; color: white; font-size: 14px; padding: 5px; border-radius: 3px;');
